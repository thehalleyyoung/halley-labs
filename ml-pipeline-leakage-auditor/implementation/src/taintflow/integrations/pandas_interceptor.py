"""
taintflow.integrations.pandas_interceptor – Pandas integration for provenance tracking.

Provides :class:`AuditedDataFrame` and :class:`AuditedSeries` wrappers that
track column-level provenance through every pandas operation, including
arithmetic, string, datetime, merge/join/concat, groupby, apply/transform,
indexing/slicing, sorting, filtering, missing-value handling, and reshaping.

Usage::

    from taintflow.integrations.pandas_interceptor import AuditedDataFrame
    df = AuditedDataFrame(raw_df, origin=Origin.TRAIN)
    result = df.merge(other_df, on="id")
    print(result.provenance)   # column-level lineage available
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    TYPE_CHECKING,
)

from taintflow.core.types import OpType, Origin, ShapeMetadata, ProvenanceInfo
from taintflow.dag.nodes import DAGNode, SourceLocation, NodeFactory

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np

try:
    import pandas as _pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===================================================================
#  Provenance annotation
# ===================================================================


@dataclass
class ProvenanceAnnotation:
    """Metadata attached to an :class:`AuditedDataFrame` or :class:`AuditedSeries`.

    Carries the data-partition origin, column-level provenance, and a
    reference to the audit session that produced it.

    Attributes:
        origin: Primary partition origin of this data.
        column_origins: Per-column origin mapping (column name → set of origins).
        source_id: Identifier linking back to the original data source.
        description: Free-form description of the data provenance.
        parent_ids: IDs of parent DataFrames that contributed.
        creation_op: The :class:`OpType` that created this annotated data.
        test_fraction: Fraction of rows originating from test partition.
    """

    origin: Origin = Origin.TRAIN
    column_origins: Dict[str, Set[Origin]] = field(default_factory=dict)
    source_id: str = ""
    description: str = ""
    parent_ids: List[str] = field(default_factory=list)
    creation_op: OpType = OpType.UNKNOWN
    test_fraction: float = 0.0

    def to_provenance_info(self) -> ProvenanceInfo:
        """Convert to a :class:`ProvenanceInfo` instance."""
        all_origins: Set[Origin] = {self.origin}
        for origins in self.column_origins.values():
            all_origins.update(origins)
        return ProvenanceInfo(
            test_fraction=self.test_fraction,
            origin_set=frozenset(all_origins),
            source_id=self.source_id,
            description=self.description,
        )

    def merge_with(self, other: "ProvenanceAnnotation") -> "ProvenanceAnnotation":
        """Combine two annotations (e.g. after a merge/concat)."""
        merged_col_origins: Dict[str, Set[Origin]] = {}
        for col, origins in self.column_origins.items():
            merged_col_origins[col] = set(origins)
        for col, origins in other.column_origins.items():
            if col in merged_col_origins:
                merged_col_origins[col].update(origins)
            else:
                merged_col_origins[col] = set(origins)

        all_parents = list(self.parent_ids) + list(other.parent_ids)
        if self.source_id:
            all_parents.append(self.source_id)
        if other.source_id:
            all_parents.append(other.source_id)

        merged_origins = {self.origin, other.origin}
        primary = Origin.TRAIN
        if Origin.TEST in merged_origins:
            primary = Origin.TEST if Origin.TRAIN not in merged_origins else Origin.TRAIN

        avg_frac = (self.test_fraction + other.test_fraction) / 2.0

        return ProvenanceAnnotation(
            origin=primary,
            column_origins=merged_col_origins,
            source_id=f"{self.source_id}+{other.source_id}",
            description=f"merged({self.description}, {other.description})",
            parent_ids=all_parents,
            creation_op=OpType.MERGE,
            test_fraction=avg_frac,
        )

    def subset_columns(self, columns: List[str]) -> "ProvenanceAnnotation":
        """Return a new annotation restricted to *columns*."""
        sub_origins = {c: set(self.column_origins[c]) for c in columns if c in self.column_origins}
        return ProvenanceAnnotation(
            origin=self.origin,
            column_origins=sub_origins,
            source_id=self.source_id,
            description=f"subset({self.description})",
            parent_ids=[self.source_id] if self.source_id else [],
            creation_op=OpType.GETITEM,
            test_fraction=self.test_fraction,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "origin": self.origin.value,
            "column_origins": {c: sorted(o.value for o in origins) for c, origins in self.column_origins.items()},
            "source_id": self.source_id,
            "description": self.description,
            "parent_ids": self.parent_ids,
            "creation_op": self.creation_op.value,
            "test_fraction": self.test_fraction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceAnnotation":
        """Deserialize from a dictionary."""
        col_origins: Dict[str, Set[Origin]] = {}
        for col, origins in data.get("column_origins", {}).items():
            col_origins[col] = {Origin.from_str(o) for o in origins}
        return cls(
            origin=Origin.from_str(data.get("origin", "train")),
            column_origins=col_origins,
            source_id=str(data.get("source_id", "")),
            description=str(data.get("description", "")),
            parent_ids=list(data.get("parent_ids", [])),
            creation_op=OpType(data.get("creation_op", "unknown")),
            test_fraction=float(data.get("test_fraction", 0.0)),
        )

    def __repr__(self) -> str:
        n_cols = len(self.column_origins)
        return (
            f"ProvenanceAnnotation(origin={self.origin.name}, "
            f"columns={n_cols}, ρ={self.test_fraction:.3f})"
        )


# ===================================================================
#  Column lineage
# ===================================================================


@dataclass
class ColumnLineage:
    """Full lineage record for a single DataFrame column.

    Tracks every operation that contributed to the column's current state,
    from the original data source through all transformations.

    Attributes:
        column_name: Name of the column.
        current_origin: Current set of origins contributing to this column.
        operations: Ordered list of (OpType, description) tuples.
        source_columns: Original column names this column derives from.
        rename_history: List of (old_name, new_name) renames applied.
    """

    column_name: str
    current_origin: Set[Origin] = field(default_factory=lambda: {Origin.TRAIN})
    operations: List[Tuple[OpType, str]] = field(default_factory=list)
    source_columns: Set[str] = field(default_factory=set)
    rename_history: List[Tuple[str, str]] = field(default_factory=list)

    def add_operation(self, op: OpType, description: str) -> None:
        """Record that *op* was applied to this column."""
        self.operations.append((op, description))

    def add_rename(self, old_name: str, new_name: str) -> None:
        """Record a rename from *old_name* to *new_name*."""
        self.rename_history.append((old_name, new_name))
        self.column_name = new_name

    def merge_lineage(self, other: "ColumnLineage") -> None:
        """Incorporate lineage information from *other*."""
        self.current_origin.update(other.current_origin)
        self.operations.extend(other.operations)
        self.source_columns.update(other.source_columns)

    @property
    def depth(self) -> int:
        """Number of operations in this column's lineage."""
        return len(self.operations)

    @property
    def is_test_tainted(self) -> bool:
        """Return *True* if test data contributed to this column."""
        return Origin.TEST in self.current_origin

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "current_origin": sorted(o.value for o in self.current_origin),
            "operations": [(op.value, desc) for op, desc in self.operations],
            "source_columns": sorted(self.source_columns),
            "rename_history": self.rename_history,
        }

    def __repr__(self) -> str:
        origins = ",".join(o.name for o in sorted(self.current_origin, key=lambda o: o.value))
        return f"ColumnLineage({self.column_name!r}, origins={{{origins}}}, depth={self.depth})"


# ===================================================================
#  Operation log
# ===================================================================


@dataclass
class _OperationEntry:
    """A single entry in the operation log."""

    op_type: OpType
    method_name: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    columns_in: List[str]
    columns_out: List[str]
    wall_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OperationLog:
    """Chronological log of all DataFrame operations in an audit session.

    Records every pandas operation together with its column inputs/outputs,
    shapes, and timings.

    Attributes:
        session_id: Unique identifier for this log session.
        entries: Ordered list of operation entries.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:12]
        self._entries: List[_OperationEntry] = []

    def record(
        self,
        op_type: OpType,
        method_name: str,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        columns_in: List[str],
        columns_out: List[str],
        wall_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an operation to the log."""
        entry = _OperationEntry(
            op_type=op_type,
            method_name=method_name,
            input_shape=input_shape,
            output_shape=output_shape,
            columns_in=columns_in,
            columns_out=columns_out,
            wall_time_ms=wall_time_ms,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.debug(
            "PandasOp: %s  in=%s out=%s  cols_in=%d cols_out=%d",
            method_name,
            input_shape,
            output_shape,
            len(columns_in),
            len(columns_out),
        )

    @property
    def entries(self) -> List[_OperationEntry]:
        """Return all entries in chronological order."""
        return list(self._entries)

    def filter_by_op(self, op_type: OpType) -> List[_OperationEntry]:
        """Return entries matching *op_type*."""
        return [e for e in self._entries if e.op_type == op_type]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def to_dag_nodes(self) -> List[DAGNode]:
        """Convert log entries to DAG nodes."""
        nodes: List[DAGNode] = []
        for entry in self._entries:
            shape = ShapeMetadata(
                n_rows=entry.output_shape[0],
                n_cols=entry.output_shape[1],
            )
            node = NodeFactory.create(op_type=entry.op_type, shape=shape)
            nodes.append(node)
        return nodes

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"OperationLog({len(self._entries)} entries, session={self.session_id!r})"


# Module-level default operation log
_default_op_log = OperationLog()


# ===================================================================
#  Helpers
# ===================================================================


def _df_shape(df: Any) -> Tuple[int, int]:
    """Extract (n_rows, n_cols) from a DataFrame-like object."""
    if hasattr(df, "shape"):
        shape = df.shape
        return (int(shape[0]), int(shape[1]) if len(shape) >= 2 else 1)
    return (0, 0)


def _df_columns(df: Any) -> List[str]:
    """Extract column names from a DataFrame-like object."""
    if hasattr(df, "columns"):
        return list(df.columns)
    return []


def _generate_df_id() -> str:
    """Generate a unique DataFrame identifier."""
    return f"df_{uuid.uuid4().hex[:10]}"


def _infer_op_type(method_name: str) -> OpType:
    """Map a pandas method name to its OpType."""
    _MAP: Dict[str, OpType] = {
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
        "isna": OpType.ISNA,
        "notna": OpType.NOTNA,
        "filter": OpType.FILTER,
        "where": OpType.WHERE,
        "mask": OpType.MASK,
        "nlargest": OpType.NLARGEST,
        "nsmallest": OpType.NSMALLEST,
        "rank": OpType.RANK,
        "diff": OpType.DIFF,
        "pct_change": OpType.PCT_CHANGE,
        "cumsum": OpType.CUMSUM,
        "cumprod": OpType.CUMPROD,
        "cummax": OpType.CUMMAX,
        "cummin": OpType.CUMMIN,
        "__getitem__": OpType.GETITEM,
        "__setitem__": OpType.SETITEM,
        "__add__": OpType.CUSTOM,
        "__sub__": OpType.CUSTOM,
        "__mul__": OpType.CUSTOM,
        "__truediv__": OpType.CUSTOM,
    }
    return _MAP.get(method_name, OpType.UNKNOWN)


# ===================================================================
#  AuditedSeries
# ===================================================================


class AuditedSeries:
    """Wraps a pandas :class:`~pandas.Series` with provenance tracking.

    All operations return a new :class:`AuditedSeries` with updated
    provenance annotations.

    Parameters
    ----------
    series
        The raw pandas Series (or any Series-like object).
    origin
        Data-partition label for this Series.
    name
        Name of the Series (overrides ``series.name`` if given).
    op_log
        Shared operation log.
    provenance
        Pre-existing provenance annotation.
    """

    def __init__(
        self,
        series: Any,
        *,
        origin: Origin = Origin.TRAIN,
        name: Optional[str] = None,
        op_log: Optional[OperationLog] = None,
        provenance: Optional[ProvenanceAnnotation] = None,
    ) -> None:
        self._series = series
        self._id = _generate_df_id()
        self._op_log = op_log or _default_op_log
        self._name = name or (series.name if hasattr(series, "name") else None)

        if provenance is not None:
            self._provenance = provenance
        else:
            col_name = str(self._name) if self._name else "value"
            self._provenance = ProvenanceAnnotation(
                origin=origin,
                column_origins={col_name: {origin}},
                source_id=self._id,
                creation_op=OpType.IDENTITY,
                test_fraction=1.0 if origin == Origin.TEST else 0.0,
            )

    @property
    def series(self) -> Any:
        """Return the underlying pandas Series."""
        return self._series

    @property
    def provenance(self) -> ProvenanceAnnotation:
        """Return the provenance annotation."""
        return self._provenance

    @property
    def name(self) -> Optional[str]:
        """Return the Series name."""
        return self._name

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying Series."""
        attr = getattr(self._series, name)
        if callable(attr):
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                result = attr(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000.0
                op = _infer_op_type(name)
                self._op_log.record(
                    op_type=op,
                    method_name=name,
                    input_shape=(len(self._series), 1),
                    output_shape=(len(result), 1) if hasattr(result, "__len__") else (0, 0),
                    columns_in=[str(self._name)] if self._name else [],
                    columns_out=[str(self._name)] if self._name else [],
                    wall_time_ms=elapsed,
                )
                if _PANDAS_AVAILABLE and isinstance(result, _pd.Series):
                    return AuditedSeries(
                        result,
                        origin=self._provenance.origin,
                        op_log=self._op_log,
                        provenance=ProvenanceAnnotation(
                            origin=self._provenance.origin,
                            column_origins=dict(self._provenance.column_origins),
                            source_id=self._id,
                            creation_op=op,
                            test_fraction=self._provenance.test_fraction,
                            parent_ids=[self._id],
                        ),
                    )
                return result
            return _wrapper
        return attr

    def __repr__(self) -> str:
        return f"AuditedSeries(name={self._name!r}, origin={self._provenance.origin.name})"

    def __len__(self) -> int:
        return len(self._series)

    # -- arithmetic operations ------------------------------------------------

    def _arith(self, other: Any, op_name: str) -> "AuditedSeries":
        """Perform an arithmetic operation and track provenance."""
        t0 = time.perf_counter()
        if isinstance(other, AuditedSeries):
            result = getattr(self._series, op_name)(other._series)
            merged_prov = self._provenance.merge_with(other._provenance)
        else:
            result = getattr(self._series, op_name)(other)
            merged_prov = ProvenanceAnnotation(
                origin=self._provenance.origin,
                column_origins=dict(self._provenance.column_origins),
                source_id=self._id,
                creation_op=OpType.CUSTOM,
                test_fraction=self._provenance.test_fraction,
                parent_ids=[self._id],
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._op_log.record(
            op_type=OpType.CUSTOM,
            method_name=op_name,
            input_shape=(len(self._series), 1),
            output_shape=(len(result), 1) if hasattr(result, "__len__") else (0, 0),
            columns_in=[str(self._name)] if self._name else [],
            columns_out=[str(self._name)] if self._name else [],
            wall_time_ms=elapsed,
        )
        return AuditedSeries(
            result, origin=merged_prov.origin, op_log=self._op_log, provenance=merged_prov,
        )

    def __add__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__add__")

    def __sub__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__sub__")

    def __mul__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__mul__")

    def __truediv__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__truediv__")

    def __radd__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__radd__")

    def __rsub__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__rsub__")

    def __rmul__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__rmul__")

    def __rtruediv__(self, other: Any) -> "AuditedSeries":
        return self._arith(other, "__rtruediv__")


# ===================================================================
#  AuditedDataFrame
# ===================================================================


class AuditedDataFrame:
    """Wraps a pandas :class:`~pandas.DataFrame` with column-level provenance.

    Intercepts every pandas operation and records it in the operation log
    while maintaining per-column provenance annotations.

    Parameters
    ----------
    df
        The raw pandas DataFrame.
    origin
        Data-partition label for this DataFrame.
    op_log
        Shared operation log; defaults to the module-level log.
    provenance
        Pre-existing provenance annotation.
    column_lineages
        Pre-existing column lineage records.
    """

    def __init__(
        self,
        df: Any,
        *,
        origin: Origin = Origin.TRAIN,
        op_log: Optional[OperationLog] = None,
        provenance: Optional[ProvenanceAnnotation] = None,
        column_lineages: Optional[Dict[str, ColumnLineage]] = None,
    ) -> None:
        self._df = df
        self._id = _generate_df_id()
        self._op_log = op_log or _default_op_log

        columns = _df_columns(df)
        if provenance is not None:
            self._provenance = provenance
        else:
            col_origins = {col: {origin} for col in columns}
            self._provenance = ProvenanceAnnotation(
                origin=origin,
                column_origins=col_origins,
                source_id=self._id,
                creation_op=OpType.IDENTITY,
                test_fraction=1.0 if origin == Origin.TEST else 0.0,
            )

        if column_lineages is not None:
            self._lineages = dict(column_lineages)
        else:
            self._lineages: Dict[str, ColumnLineage] = {}
            for col in columns:
                self._lineages[col] = ColumnLineage(
                    column_name=col,
                    current_origin={origin},
                    source_columns={col},
                )

    @property
    def df(self) -> Any:
        """Return the underlying pandas DataFrame."""
        return self._df

    @property
    def provenance(self) -> ProvenanceAnnotation:
        """Return the provenance annotation."""
        return self._provenance

    @property
    def column_lineages(self) -> Dict[str, ColumnLineage]:
        """Return per-column lineage records."""
        return dict(self._lineages)

    @property
    def columns(self) -> List[str]:
        """Return the column names."""
        return _df_columns(self._df)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return ``(n_rows, n_cols)``."""
        return _df_shape(self._df)

    @property
    def dtypes(self) -> Any:
        """Return column dtypes."""
        return self._df.dtypes if hasattr(self._df, "dtypes") else None

    # -- internal helpers -----------------------------------------------------

    def _make_child(
        self,
        result_df: Any,
        op_type: OpType,
        method_name: str,
        wall_time_ms: float,
        *,
        extra_provenance: Optional[ProvenanceAnnotation] = None,
        dropped_cols: Optional[List[str]] = None,
        added_cols: Optional[List[str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
    ) -> "AuditedDataFrame":
        """Wrap *result_df* in a new :class:`AuditedDataFrame` with updated provenance."""
        in_shape = _df_shape(self._df)
        out_shape = _df_shape(result_df)
        in_cols = _df_columns(self._df)
        out_cols = _df_columns(result_df)

        self._op_log.record(
            op_type=op_type,
            method_name=method_name,
            input_shape=in_shape,
            output_shape=out_shape,
            columns_in=in_cols,
            columns_out=out_cols,
            wall_time_ms=wall_time_ms,
        )

        # Build new provenance
        if extra_provenance is not None:
            new_prov = self._provenance.merge_with(extra_provenance)
            new_prov.creation_op = op_type
        else:
            new_col_origins: Dict[str, Set[Origin]] = {}
            for col in out_cols:
                if rename_map and col in rename_map.values():
                    old = next((k for k, v in rename_map.items() if v == col), col)
                    new_col_origins[col] = set(self._provenance.column_origins.get(old, {self._provenance.origin}))
                elif col in self._provenance.column_origins:
                    new_col_origins[col] = set(self._provenance.column_origins[col])
                else:
                    new_col_origins[col] = {self._provenance.origin}
            new_prov = ProvenanceAnnotation(
                origin=self._provenance.origin,
                column_origins=new_col_origins,
                source_id=self._id,
                description=f"{method_name}({self._provenance.description})",
                parent_ids=[self._id],
                creation_op=op_type,
                test_fraction=self._provenance.test_fraction,
            )

        # Build new lineages
        new_lineages: Dict[str, ColumnLineage] = {}
        for col in out_cols:
            if rename_map and col in rename_map.values():
                old = next((k for k, v in rename_map.items() if v == col), col)
                if old in self._lineages:
                    lineage = copy.deepcopy(self._lineages[old])
                    lineage.add_rename(old, col)
                    lineage.add_operation(op_type, method_name)
                    new_lineages[col] = lineage
                else:
                    new_lineages[col] = ColumnLineage(
                        column_name=col,
                        current_origin={self._provenance.origin},
                        source_columns={col},
                        operations=[(op_type, method_name)],
                    )
            elif col in self._lineages:
                lineage = copy.deepcopy(self._lineages[col])
                lineage.add_operation(op_type, method_name)
                new_lineages[col] = lineage
            else:
                new_lineages[col] = ColumnLineage(
                    column_name=col,
                    current_origin={self._provenance.origin},
                    source_columns={col},
                    operations=[(op_type, method_name)],
                )

        return AuditedDataFrame(
            result_df,
            origin=new_prov.origin,
            op_log=self._op_log,
            provenance=new_prov,
            column_lineages=new_lineages,
        )

    # -- delegation for unknown methods ---------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute access to the underlying DataFrame."""
        attr = getattr(self._df, name)
        if callable(attr):
            def _delegated(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                result = attr(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000.0
                op = _infer_op_type(name)
                if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
                    return self._make_child(result, op, name, elapsed)
                if _PANDAS_AVAILABLE and isinstance(result, _pd.Series):
                    col_name = result.name if hasattr(result, "name") else name
                    return AuditedSeries(
                        result,
                        origin=self._provenance.origin,
                        name=str(col_name) if col_name is not None else None,
                        op_log=self._op_log,
                    )
                return result
            return _delegated
        return attr

    def __repr__(self) -> str:
        rows, cols = self.shape
        return (
            f"AuditedDataFrame({rows}×{cols}, "
            f"origin={self._provenance.origin.name})"
        )

    def __len__(self) -> int:
        return len(self._df)

    # -- indexing / slicing ---------------------------------------------------

    def __getitem__(self, key: Any) -> Union["AuditedDataFrame", AuditedSeries]:
        """Column selection or boolean indexing with provenance tracking."""
        t0 = time.perf_counter()
        result = self._df[key]
        elapsed = (time.perf_counter() - t0) * 1000.0

        if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
            selected_cols = list(key) if isinstance(key, (list, tuple)) else _df_columns(result)
            sub_prov = self._provenance.subset_columns(selected_cols)
            return self._make_child(result, OpType.GETITEM, "__getitem__", elapsed)
        if _PANDAS_AVAILABLE and isinstance(result, _pd.Series):
            col_name = key if isinstance(key, str) else (result.name if hasattr(result, "name") else None)
            self._op_log.record(
                op_type=OpType.GETITEM,
                method_name="__getitem__",
                input_shape=_df_shape(self._df),
                output_shape=(len(result), 1),
                columns_in=_df_columns(self._df),
                columns_out=[str(col_name)] if col_name else [],
                wall_time_ms=elapsed,
            )
            return AuditedSeries(
                result,
                origin=self._provenance.origin,
                name=str(col_name) if col_name is not None else None,
                op_log=self._op_log,
            )
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        """Column assignment with provenance tracking."""
        t0 = time.perf_counter()
        raw_value = value._series if isinstance(value, AuditedSeries) else value
        self._df[key] = raw_value
        elapsed = (time.perf_counter() - t0) * 1000.0

        col_name = str(key) if isinstance(key, str) else str(key)
        self._provenance.column_origins[col_name] = {self._provenance.origin}
        if isinstance(value, AuditedSeries):
            self._provenance.column_origins[col_name].update(
                value.provenance.column_origins.get(str(value.name), set()),
            )

        self._lineages[col_name] = ColumnLineage(
            column_name=col_name,
            current_origin=set(self._provenance.column_origins.get(col_name, {self._provenance.origin})),
            operations=[(OpType.SETITEM, f"__setitem__({col_name})")],
            source_columns={col_name},
        )

        self._op_log.record(
            op_type=OpType.SETITEM,
            method_name="__setitem__",
            input_shape=_df_shape(self._df),
            output_shape=_df_shape(self._df),
            columns_in=_df_columns(self._df),
            columns_out=_df_columns(self._df),
            wall_time_ms=elapsed,
        )

    # -- merge / join / concat ------------------------------------------------

    def merge(self, right: Any, **kwargs: Any) -> "AuditedDataFrame":
        """Merge with another DataFrame, tracking provenance from both sides."""
        t0 = time.perf_counter()
        right_df = right._df if isinstance(right, AuditedDataFrame) else right
        result = self._df.merge(right_df, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0

        right_prov = right._provenance if isinstance(right, AuditedDataFrame) else None
        return self._make_child(result, OpType.MERGE, "merge", elapsed, extra_provenance=right_prov)

    def join(self, other: Any, **kwargs: Any) -> "AuditedDataFrame":
        """Join with another DataFrame, tracking provenance."""
        t0 = time.perf_counter()
        other_df = other._df if isinstance(other, AuditedDataFrame) else other
        result = self._df.join(other_df, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0

        other_prov = other._provenance if isinstance(other, AuditedDataFrame) else None
        return self._make_child(result, OpType.JOIN, "join", elapsed, extra_provenance=other_prov)

    # -- groupby ---------------------------------------------------------------

    def groupby(self, by: Any, **kwargs: Any) -> "_AuditedGroupBy":
        """Group by columns, returning an audited GroupBy proxy."""
        self._op_log.record(
            op_type=OpType.GROUPBY,
            method_name="groupby",
            input_shape=_df_shape(self._df),
            output_shape=_df_shape(self._df),
            columns_in=_df_columns(self._df),
            columns_out=_df_columns(self._df),
            wall_time_ms=0.0,
        )
        return _AuditedGroupBy(
            groupby_obj=self._df.groupby(by, **kwargs),
            parent=self,
            group_cols=by if isinstance(by, list) else [by],
        )

    # -- missing values -------------------------------------------------------

    def fillna(self, value: Any = None, **kwargs: Any) -> "AuditedDataFrame":
        """Fill missing values, tracking the operation."""
        t0 = time.perf_counter()
        result = self._df.fillna(value, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.FILLNA, "fillna", elapsed)

    def dropna(self, **kwargs: Any) -> "AuditedDataFrame":
        """Drop rows/columns with missing values."""
        t0 = time.perf_counter()
        result = self._df.dropna(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.DROPNA, "dropna", elapsed)

    def interpolate(self, **kwargs: Any) -> "AuditedDataFrame":
        """Interpolate missing values."""
        t0 = time.perf_counter()
        result = self._df.interpolate(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.INTERPOLATE, "interpolate", elapsed)

    # -- sorting / filtering --------------------------------------------------

    def sort_values(self, by: Any, **kwargs: Any) -> "AuditedDataFrame":
        """Sort by values, tracking the operation."""
        t0 = time.perf_counter()
        result = self._df.sort_values(by, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.SORT_VALUES, "sort_values", elapsed)

    def sort_index(self, **kwargs: Any) -> "AuditedDataFrame":
        """Sort by index."""
        t0 = time.perf_counter()
        result = self._df.sort_index(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.SORT_INDEX, "sort_index", elapsed)

    def query(self, expr: str, **kwargs: Any) -> "AuditedDataFrame":
        """Filter rows by expression."""
        t0 = time.perf_counter()
        result = self._df.query(expr, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.QUERY, "query", elapsed)

    # -- reshaping ------------------------------------------------------------

    def pivot(self, **kwargs: Any) -> "AuditedDataFrame":
        """Pivot the DataFrame."""
        t0 = time.perf_counter()
        result = self._df.pivot(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.PIVOT, "pivot", elapsed)

    def pivot_table(self, **kwargs: Any) -> "AuditedDataFrame":
        """Pivot table aggregation."""
        t0 = time.perf_counter()
        result = self._df.pivot_table(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.PIVOT_TABLE, "pivot_table", elapsed)

    def melt(self, **kwargs: Any) -> "AuditedDataFrame":
        """Unpivot (melt) the DataFrame."""
        t0 = time.perf_counter()
        result = self._df.melt(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.MELT, "melt", elapsed)

    def stack(self, **kwargs: Any) -> Any:
        """Stack columns into rows."""
        t0 = time.perf_counter()
        result = self._df.stack(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
            return self._make_child(result, OpType.STACK, "stack", elapsed)
        if _PANDAS_AVAILABLE and isinstance(result, _pd.Series):
            self._op_log.record(
                op_type=OpType.STACK,
                method_name="stack",
                input_shape=_df_shape(self._df),
                output_shape=(len(result), 1),
                columns_in=_df_columns(self._df),
                columns_out=[str(result.name)] if result.name else [],
                wall_time_ms=elapsed,
            )
            return AuditedSeries(result, origin=self._provenance.origin, op_log=self._op_log)
        return result

    def unstack(self, **kwargs: Any) -> "AuditedDataFrame":
        """Unstack rows into columns."""
        t0 = time.perf_counter()
        result = self._df.unstack(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
            return self._make_child(result, OpType.UNSTACK, "unstack", elapsed)
        return result  # type: ignore[return-value]

    # -- column manipulation ---------------------------------------------------

    def drop(self, labels: Any = None, **kwargs: Any) -> "AuditedDataFrame":
        """Drop columns or rows."""
        t0 = time.perf_counter()
        result = self._df.drop(labels, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        dropped = labels if isinstance(labels, list) else ([labels] if labels is not None else [])
        return self._make_child(
            result, OpType.DROP, "drop", elapsed, dropped_cols=dropped,
        )

    def rename(self, columns: Optional[Dict[str, str]] = None, **kwargs: Any) -> "AuditedDataFrame":
        """Rename columns with lineage tracking."""
        t0 = time.perf_counter()
        result = self._df.rename(columns=columns, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(
            result, OpType.RENAME, "rename", elapsed, rename_map=columns,
        )

    def assign(self, **kwargs: Any) -> "AuditedDataFrame":
        """Assign new columns."""
        t0 = time.perf_counter()
        result = self._df.assign(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(
            result, OpType.ASSIGN, "assign", elapsed, added_cols=list(kwargs.keys()),
        )

    def drop_duplicates(self, **kwargs: Any) -> "AuditedDataFrame":
        """Drop duplicate rows."""
        t0 = time.perf_counter()
        result = self._df.drop_duplicates(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.DROP_DUPLICATES, "drop_duplicates", elapsed)

    def set_index(self, keys: Any, **kwargs: Any) -> "AuditedDataFrame":
        """Set the index."""
        t0 = time.perf_counter()
        result = self._df.set_index(keys, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.SET_INDEX, "set_index", elapsed)

    def reset_index(self, **kwargs: Any) -> "AuditedDataFrame":
        """Reset the index."""
        t0 = time.perf_counter()
        result = self._df.reset_index(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.RESET_INDEX, "reset_index", elapsed)

    # -- apply / transform ----------------------------------------------------

    def apply(self, func: Callable[..., Any], **kwargs: Any) -> Any:
        """Apply a function along an axis, tracking the operation."""
        t0 = time.perf_counter()
        result = self._df.apply(func, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
            return self._make_child(result, OpType.APPLY, "apply", elapsed)
        if _PANDAS_AVAILABLE and isinstance(result, _pd.Series):
            self._op_log.record(
                op_type=OpType.APPLY,
                method_name="apply",
                input_shape=_df_shape(self._df),
                output_shape=(len(result), 1),
                columns_in=_df_columns(self._df),
                columns_out=[str(result.name)] if result.name else [],
                wall_time_ms=elapsed,
            )
            return AuditedSeries(result, origin=self._provenance.origin, op_log=self._op_log)
        return result

    # -- string / datetime / categorical accessors ----------------------------

    @property
    def str(self) -> Any:
        """Delegate ``str`` accessor, logging the access."""
        self._op_log.record(
            op_type=OpType.STR_ACCESSOR,
            method_name="str",
            input_shape=_df_shape(self._df),
            output_shape=_df_shape(self._df),
            columns_in=_df_columns(self._df),
            columns_out=_df_columns(self._df),
        )
        return self._df.str if hasattr(self._df, "str") else None

    @property
    def dt(self) -> Any:
        """Delegate ``dt`` accessor, logging the access."""
        self._op_log.record(
            op_type=OpType.DT_ACCESSOR,
            method_name="dt",
            input_shape=_df_shape(self._df),
            output_shape=_df_shape(self._df),
            columns_in=_df_columns(self._df),
            columns_out=_df_columns(self._df),
        )
        return self._df.dt if hasattr(self._df, "dt") else None

    @property
    def cat(self) -> Any:
        """Delegate ``cat`` accessor, logging the access."""
        self._op_log.record(
            op_type=OpType.CAT_ACCESSOR,
            method_name="cat",
            input_shape=_df_shape(self._df),
            output_shape=_df_shape(self._df),
            columns_in=_df_columns(self._df),
            columns_out=_df_columns(self._df),
        )
        return self._df.cat if hasattr(self._df, "cat") else None

    # -- arithmetic operations ------------------------------------------------

    def _arith_df(self, other: Any, op_name: str) -> "AuditedDataFrame":
        """Perform arithmetic between DataFrames with provenance tracking."""
        t0 = time.perf_counter()
        if isinstance(other, AuditedDataFrame):
            result = getattr(self._df, op_name)(other._df)
            merged_prov = self._provenance.merge_with(other._provenance)
        else:
            result = getattr(self._df, op_name)(other)
            merged_prov = ProvenanceAnnotation(
                origin=self._provenance.origin,
                column_origins=dict(self._provenance.column_origins),
                source_id=self._id,
                creation_op=OpType.CUSTOM,
                test_fraction=self._provenance.test_fraction,
                parent_ids=[self._id],
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
            return self._make_child(result, OpType.CUSTOM, op_name, elapsed, extra_provenance=merged_prov)
        return self._make_child(result, OpType.CUSTOM, op_name, elapsed)

    def __add__(self, other: Any) -> "AuditedDataFrame":
        return self._arith_df(other, "__add__")

    def __sub__(self, other: Any) -> "AuditedDataFrame":
        return self._arith_df(other, "__sub__")

    def __mul__(self, other: Any) -> "AuditedDataFrame":
        return self._arith_df(other, "__mul__")

    def __truediv__(self, other: Any) -> "AuditedDataFrame":
        return self._arith_df(other, "__truediv__")

    # -- copy -----------------------------------------------------------------

    def copy(self, deep: bool = True) -> "AuditedDataFrame":
        """Copy the DataFrame with provenance preserved."""
        t0 = time.perf_counter()
        result = self._df.copy(deep=deep)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.COPY, "copy", elapsed)

    # -- astype ---------------------------------------------------------------

    def astype(self, dtype: Any, **kwargs: Any) -> "AuditedDataFrame":
        """Cast dtypes."""
        t0 = time.perf_counter()
        result = self._df.astype(dtype, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.ASTYPE, "astype", elapsed)

    # -- replace / clip -------------------------------------------------------

    def replace(self, to_replace: Any = None, value: Any = None, **kwargs: Any) -> "AuditedDataFrame":
        """Replace values."""
        t0 = time.perf_counter()
        result = self._df.replace(to_replace, value, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.REPLACE, "replace", elapsed)

    def clip(self, lower: Any = None, upper: Any = None, **kwargs: Any) -> "AuditedDataFrame":
        """Clip values."""
        t0 = time.perf_counter()
        result = self._df.clip(lower, upper, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.CLIP, "clip", elapsed)

    # -- head / tail / sample --------------------------------------------------

    def head(self, n: int = 5) -> "AuditedDataFrame":
        """Return first *n* rows."""
        t0 = time.perf_counter()
        result = self._df.head(n)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.HEAD, "head", elapsed)

    def tail(self, n: int = 5) -> "AuditedDataFrame":
        """Return last *n* rows."""
        t0 = time.perf_counter()
        result = self._df.tail(n)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.TAIL, "tail", elapsed)

    def sample(self, **kwargs: Any) -> "AuditedDataFrame":
        """Random sample of rows."""
        t0 = time.perf_counter()
        result = self._df.sample(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._make_child(result, OpType.SAMPLE, "sample", elapsed)


# ===================================================================
#  Audited GroupBy proxy
# ===================================================================


class _AuditedGroupBy:
    """Proxy for a pandas GroupBy object that tracks provenance.

    Intercepts aggregation, transform, and apply calls so that the
    resulting DataFrame / Series carries updated provenance.
    """

    def __init__(
        self,
        groupby_obj: Any,
        parent: AuditedDataFrame,
        group_cols: List[str],
    ) -> None:
        self._groupby = groupby_obj
        self._parent = parent
        self._group_cols = group_cols

    def _wrap_result(self, result: Any, op: OpType, method: str, elapsed: float) -> Any:
        if _PANDAS_AVAILABLE and isinstance(result, _pd.DataFrame):
            return self._parent._make_child(result, op, f"groupby.{method}", elapsed)
        if _PANDAS_AVAILABLE and isinstance(result, _pd.Series):
            self._parent._op_log.record(
                op_type=op,
                method_name=f"groupby.{method}",
                input_shape=_df_shape(self._parent._df),
                output_shape=(len(result), 1),
                columns_in=_df_columns(self._parent._df),
                columns_out=[str(result.name)] if result.name else [],
                wall_time_ms=elapsed,
            )
            return AuditedSeries(
                result, origin=self._parent._provenance.origin, op_log=self._parent._op_log,
            )
        return result

    def agg(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Aggregate groups."""
        t0 = time.perf_counter()
        result = self._groupby.agg(func, *args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "agg", elapsed)

    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Aggregate groups (alias for ``agg``)."""
        t0 = time.perf_counter()
        result = self._groupby.aggregate(func, *args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGGREGATE, "aggregate", elapsed)

    def transform(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform groups."""
        t0 = time.perf_counter()
        result = self._groupby.transform(func, *args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.TRANSFORM, "transform", elapsed)

    def apply(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply function to each group."""
        t0 = time.perf_counter()
        result = self._groupby.apply(func, *args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.APPLY, "apply", elapsed)

    def mean(self, **kwargs: Any) -> Any:
        """Compute group means."""
        t0 = time.perf_counter()
        result = self._groupby.mean(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "mean", elapsed)

    def sum(self, **kwargs: Any) -> Any:
        """Compute group sums."""
        t0 = time.perf_counter()
        result = self._groupby.sum(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "sum", elapsed)

    def std(self, **kwargs: Any) -> Any:
        """Compute group standard deviations."""
        t0 = time.perf_counter()
        result = self._groupby.std(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "std", elapsed)

    def count(self, **kwargs: Any) -> Any:
        """Compute group counts."""
        t0 = time.perf_counter()
        result = self._groupby.count(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "count", elapsed)

    def median(self, **kwargs: Any) -> Any:
        """Compute group medians."""
        t0 = time.perf_counter()
        result = self._groupby.median(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "median", elapsed)

    def min(self, **kwargs: Any) -> Any:
        """Compute group minimums."""
        t0 = time.perf_counter()
        result = self._groupby.min(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "min", elapsed)

    def max(self, **kwargs: Any) -> Any:
        """Compute group maximums."""
        t0 = time.perf_counter()
        result = self._groupby.max(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._wrap_result(result, OpType.AGG, "max", elapsed)

    def __getattr__(self, name: str) -> Any:
        """Delegate to the underlying GroupBy for unhandled methods."""
        return getattr(self._groupby, name)


# ===================================================================
#  DataFrameAuditor – audit operations on existing DataFrames
# ===================================================================


class DataFrameAuditor:
    """Audit operations on existing (unmodified) pandas DataFrames.

    Unlike :class:`AuditedDataFrame` which wraps a DataFrame at creation,
    ``DataFrameAuditor`` can attach to an already-existing DataFrame and
    retrospectively build provenance annotations from an operation log.

    Parameters
    ----------
    op_log
        The operation log to use for recording.
    default_origin
        Default partition origin for untagged DataFrames.
    """

    def __init__(
        self,
        op_log: Optional[OperationLog] = None,
        default_origin: Origin = Origin.TRAIN,
    ) -> None:
        self._op_log = op_log or OperationLog()
        self._default_origin = default_origin
        self._tracked: Dict[str, AuditedDataFrame] = {}

    @property
    def op_log(self) -> OperationLog:
        """Return the operation log."""
        return self._op_log

    @property
    def tracked_frames(self) -> Dict[str, AuditedDataFrame]:
        """Return all tracked DataFrames by their IDs."""
        return dict(self._tracked)

    def wrap(
        self,
        df: Any,
        *,
        origin: Optional[Origin] = None,
        name: Optional[str] = None,
    ) -> AuditedDataFrame:
        """Wrap *df* in an :class:`AuditedDataFrame` and start tracking it.

        Parameters
        ----------
        df
            A pandas DataFrame.
        origin
            Partition origin for this DataFrame.
        name
            Optional human-readable name (used in the provenance annotation).

        Returns
        -------
        AuditedDataFrame
        """
        origin = origin or self._default_origin
        audited = AuditedDataFrame(df, origin=origin, op_log=self._op_log)
        if name:
            audited._provenance.description = name
        self._tracked[audited._id] = audited
        return audited

    def build_lineage_report(self) -> Dict[str, Dict[str, ColumnLineage]]:
        """Build a lineage report for all tracked DataFrames.

        Returns
        -------
        dict
            Mapping from DataFrame ID to a dict of column lineages.
        """
        report: Dict[str, Dict[str, ColumnLineage]] = {}
        for df_id, adf in self._tracked.items():
            report[df_id] = adf.column_lineages
        return report

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all tracked DataFrames and operations."""
        return {
            "n_tracked_frames": len(self._tracked),
            "n_operations": len(self._op_log),
            "frames": {
                df_id: {
                    "shape": adf.shape,
                    "origin": adf.provenance.origin.value,
                    "n_columns": len(adf.columns),
                }
                for df_id, adf in self._tracked.items()
            },
        }

    def to_dag_nodes(self) -> List[DAGNode]:
        """Convert recorded operations to DAG nodes."""
        return self._op_log.to_dag_nodes()
