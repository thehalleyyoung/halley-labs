"""
taintflow.instrument.pandas_hooks – Monkey-patching layer for pandas
DataFrame, Series, and GroupBy methods.

Intercepts pandas operations at runtime to capture operation type, input
and output shapes, column-level changes, and row provenance bitmaps.
All hooks are installed and removed cleanly via :class:`PandasInterceptor`.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

from taintflow.core.types import OpType, Origin, ProvenanceInfo, ShapeMetadata
from taintflow.dag.nodes import DAGNode, NodeFactory, SourceLocation

logger = logging.getLogger(__name__)


# ===================================================================
#  Hook metadata
# ===================================================================

class HookStatus(Enum):
    """Installation status of a single hook."""

    PENDING = auto()
    INSTALLED = auto()
    FAILED = auto()
    REMOVED = auto()


@dataclass
class HookMetadata:
    """Metadata about a single installed hook.

    Attributes:
        hook_id: Unique hook identifier.
        target_class: Fully-qualified name of the class being patched.
        method_name: Name of the method being patched.
        op_type: The :class:`OpType` assigned to this hook.
        status: Installation status.
        installed_at: Monotonic timestamp when the hook was installed.
        removed_at: Monotonic timestamp when the hook was removed.
        call_count: Number of times the hook was invoked.
        error_count: Number of times the hook raised an exception.
        last_error: String representation of the most recent error.
    """

    hook_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    target_class: str = ""
    method_name: str = ""
    op_type: OpType = OpType.UNKNOWN
    status: HookStatus = HookStatus.PENDING
    installed_at: float = 0.0
    removed_at: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "target": f"{self.target_class}.{self.method_name}",
            "op_type": self.op_type.value,
            "status": self.status.name,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


# ===================================================================
#  Intercept record
# ===================================================================

@dataclass
class PandasInterceptRecord:
    """Record of a single intercepted pandas call.

    Attributes:
        record_id: Unique record identifier.
        timestamp: Monotonic time of the call.
        op_type: Mapped operation type.
        method_name: Name of the method that was called.
        target_class: Class of the object the method was called on.
        input_shape: (n_rows, n_cols) of the primary input.
        output_shape: (n_rows, n_cols) of the result.
        input_columns: Column names of the input DataFrame/Series.
        output_columns: Column names of the output DataFrame/Series.
        kwargs_keys: Keys of the keyword arguments passed to the method.
        wall_time_ms: Execution time of the original method in milliseconds.
        provenance: Row provenance information, if available.
        extra: Additional operation-specific metadata.
    """

    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = 0.0
    op_type: OpType = OpType.UNKNOWN
    method_name: str = ""
    target_class: str = ""
    input_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, int]] = None
    input_columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    kwargs_keys: List[str] = field(default_factory=list)
    wall_time_ms: float = 0.0
    provenance: Optional[ProvenanceInfo] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "op_type": self.op_type.value,
            "method_name": self.method_name,
            "target_class": self.target_class,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "wall_time_ms": round(self.wall_time_ms, 4),
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        if self.extra:
            d["extra"] = self.extra
        return d


# ===================================================================
#  Method -> OpType mapping
# ===================================================================

# DataFrame methods
_DF_METHOD_MAP: Dict[str, OpType] = {
    "merge": OpType.MERGE,
    "join": OpType.JOIN,
    "groupby": OpType.GROUPBY,
    "apply": OpType.APPLY,
    "transform": OpType.TRANSFORM,
    "assign": OpType.ASSIGN,
    "drop": OpType.DROP,
    "drop_duplicates": OpType.DROP_DUPLICATES,
    "rename": OpType.RENAME,
    "fillna": OpType.FILLNA,
    "dropna": OpType.DROPNA,
    "pivot": OpType.PIVOT,
    "pivot_table": OpType.PIVOT_TABLE,
    "melt": OpType.MELT,
    "stack": OpType.STACK,
    "unstack": OpType.UNSTACK,
    "explode": OpType.EXPLODE,
    "sort_values": OpType.SORT_VALUES,
    "sort_index": OpType.SORT_INDEX,
    "set_index": OpType.SET_INDEX,
    "reset_index": OpType.RESET_INDEX,
    "reindex": OpType.REINDEX,
    "sample": OpType.SAMPLE,
    "head": OpType.HEAD,
    "tail": OpType.TAIL,
    "nlargest": OpType.NLARGEST,
    "nsmallest": OpType.NSMALLEST,
    "query": OpType.QUERY,
    "where": OpType.WHERE,
    "mask": OpType.MASK,
    "clip": OpType.CLIP,
    "replace": OpType.REPLACE,
    "interpolate": OpType.INTERPOLATE,
    "astype": OpType.ASTYPE,
    "copy": OpType.COPY,
    "describe": OpType.DESCRIBE,
    "corr": OpType.CORR,
    "cov": OpType.COV,
    "value_counts": OpType.VALUE_COUNTS,
    "agg": OpType.AGG,
    "aggregate": OpType.AGGREGATE,
    "applymap": OpType.APPLYMAP,
    "rolling": OpType.ROLLING,
    "expanding": OpType.EXPANDING,
    "ewm": OpType.EWM,
    "resample": OpType.RESAMPLE,
    "diff": OpType.DIFF,
    "pct_change": OpType.PCT_CHANGE,
    "rank": OpType.RANK,
    "cumsum": OpType.CUMSUM,
    "cumprod": OpType.CUMPROD,
    "cummax": OpType.CUMMAX,
    "cummin": OpType.CUMMIN,
    "insert": OpType.INSERT,
    "pop": OpType.POP,
}

# Series methods (additional to those shared with DataFrame)
_SERIES_METHOD_MAP: Dict[str, OpType] = {
    "map": OpType.MAP,
    "apply": OpType.APPLY,
    "fillna": OpType.FILLNA,
    "dropna": OpType.DROPNA,
    "replace": OpType.REPLACE,
    "astype": OpType.ASTYPE,
    "clip": OpType.CLIP,
    "interpolate": OpType.INTERPOLATE,
    "value_counts": OpType.VALUE_COUNTS,
    "sort_values": OpType.SORT_VALUES,
    "sort_index": OpType.SORT_INDEX,
    "rank": OpType.RANK,
    "diff": OpType.DIFF,
    "pct_change": OpType.PCT_CHANGE,
    "describe": OpType.DESCRIBE,
    "sample": OpType.SAMPLE,
    "head": OpType.HEAD,
    "tail": OpType.TAIL,
    "copy": OpType.COPY,
    "rename": OpType.RENAME,
    "reset_index": OpType.RESET_INDEX,
    "explode": OpType.EXPLODE,
}

# GroupBy methods
_GROUPBY_METHOD_MAP: Dict[str, OpType] = {
    "transform": OpType.TRANSFORM,
    "aggregate": OpType.AGGREGATE,
    "agg": OpType.AGG,
    "apply": OpType.APPLY,
    "filter": OpType.FILTER,
    "describe": OpType.DESCRIBE,
    "value_counts": OpType.VALUE_COUNTS,
    "cumsum": OpType.CUMSUM,
    "cumprod": OpType.CUMPROD,
    "cummax": OpType.CUMMAX,
    "cummin": OpType.CUMMIN,
    "diff": OpType.DIFF,
    "pct_change": OpType.PCT_CHANGE,
    "rank": OpType.RANK,
    "fillna": OpType.FILLNA,
    "rolling": OpType.ROLLING,
    "expanding": OpType.EXPANDING,
    "resample": OpType.RESAMPLE,
}

# Module-level functions
_MODULE_FUNC_MAP: Dict[str, OpType] = {
    "concat": OpType.CONCAT,
    "merge": OpType.MERGE,
    "get_dummies": OpType.GET_DUMMIES,
    "crosstab": OpType.CROSSTAB,
    "cut": OpType.CUT,
    "qcut": OpType.QCUT,
    "factorize": OpType.FACTORIZE,
    "read_csv": OpType.READ_CSV,
    "read_parquet": OpType.READ_PARQUET,
    "read_json": OpType.READ_JSON,
    "read_excel": OpType.READ_EXCEL,
    "read_sql": OpType.READ_SQL,
    "read_hdf": OpType.READ_HDF,
    "read_feather": OpType.READ_FEATHER,
}


# ===================================================================
#  Shape / column helpers
# ===================================================================

def _safe_shape(obj: Any) -> Optional[Tuple[int, int]]:
    """Extract (n_rows, n_cols) from an object if it has a shape attribute."""
    try:
        shape = obj.shape
        if len(shape) == 2:
            return (int(shape[0]), int(shape[1]))
        if len(shape) == 1:
            return (int(shape[0]), 1)
    except (AttributeError, TypeError, IndexError):
        pass
    return None


def _safe_columns(obj: Any) -> List[str]:
    """Extract column names from a DataFrame or Series."""
    try:
        if hasattr(obj, "columns"):
            return [str(c) for c in obj.columns.tolist()]
        if hasattr(obj, "name") and obj.name is not None:
            return [str(obj.name)]
    except (AttributeError, TypeError):
        pass
    return []


def _safe_class_name(obj: Any) -> str:
    """Return the qualified class name of *obj*."""
    try:
        cls = type(obj)
        module = cls.__module__ or ""
        return f"{module}.{cls.__qualname__}"
    except (AttributeError, TypeError):
        return "<unknown>"


def _extract_merge_extra(kwargs: Dict[str, Any], args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Extract merge-specific metadata."""
    extra: Dict[str, Any] = {}
    extra["how"] = kwargs.get("how", "inner")
    if "on" in kwargs:
        extra["on"] = kwargs["on"] if isinstance(kwargs["on"], list) else [kwargs["on"]]
    if "left_on" in kwargs:
        extra["left_on"] = kwargs["left_on"]
    if "right_on" in kwargs:
        extra["right_on"] = kwargs["right_on"]
    if len(args) > 0:
        extra["right_shape"] = _safe_shape(args[0])
    return extra


def _extract_groupby_extra(kwargs: Dict[str, Any], args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Extract groupby-specific metadata."""
    extra: Dict[str, Any] = {}
    if "by" in kwargs:
        by = kwargs["by"]
        extra["by"] = by if isinstance(by, list) else [by]
    elif len(args) > 0:
        by = args[0]
        extra["by"] = by if isinstance(by, list) else [by]
    return extra


def _extract_concat_extra(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract concat-specific metadata."""
    extra: Dict[str, Any] = {}
    extra["axis"] = kwargs.get("axis", 0)
    objs = args[0] if args else kwargs.get("objs", [])
    try:
        extra["n_inputs"] = len(objs)
        extra["input_shapes"] = [_safe_shape(o) for o in objs]
    except TypeError:
        pass
    return extra


def _extract_read_extra(method_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract IO read-specific metadata."""
    extra: Dict[str, Any] = {}
    if args:
        source = args[0]
        if isinstance(source, str):
            extra["source_path"] = source
    elif "filepath_or_buffer" in kwargs:
        source = kwargs["filepath_or_buffer"]
        if isinstance(source, str):
            extra["source_path"] = source
    extra["reader"] = method_name
    return extra


# ===================================================================
#  HookRegistry
# ===================================================================

class HookRegistry:
    """Manages installed monkey-patch hooks and guarantees cleanup.

    Each hook is tracked by a :class:`HookMetadata` record.  The registry
    stores the original (unpatched) method reference so that it can be
    restored on :meth:`remove_all`.
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, HookMetadata] = {}
        self._originals: Dict[str, Tuple[Any, str, Any]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        target: Any,
        method_name: str,
        wrapper: Callable[..., Any],
        metadata: HookMetadata,
    ) -> bool:
        """Install *wrapper* as a replacement for *target.method_name*.

        Returns True on success.
        """
        key = f"{metadata.target_class}.{method_name}"
        with self._lock:
            if key in self._hooks:
                return False
            try:
                original = getattr(target, method_name)
                setattr(target, method_name, wrapper)
                self._originals[key] = (target, method_name, original)
                metadata.status = HookStatus.INSTALLED
                metadata.installed_at = time.monotonic()
                self._hooks[key] = metadata
                return True
            except (AttributeError, TypeError) as exc:
                metadata.status = HookStatus.FAILED
                metadata.last_error = str(exc)
                self._hooks[key] = metadata
                return False

    def remove(self, key: str) -> bool:
        """Remove a single hook by key and restore the original method."""
        with self._lock:
            if key not in self._originals:
                return False
            target, method_name, original = self._originals.pop(key)
            try:
                setattr(target, method_name, original)
            except (AttributeError, TypeError):
                pass
            if key in self._hooks:
                self._hooks[key].status = HookStatus.REMOVED
                self._hooks[key].removed_at = time.monotonic()
            return True

    def remove_all(self) -> int:
        """Remove all hooks, restoring original methods.  Returns count removed."""
        with self._lock:
            keys = list(self._originals.keys())
        removed = 0
        for key in keys:
            if self.remove(key):
                removed += 1
        return removed

    @property
    def installed_hooks(self) -> List[HookMetadata]:
        """Return metadata for currently installed hooks."""
        with self._lock:
            return [h for h in self._hooks.values() if h.status == HookStatus.INSTALLED]

    @property
    def all_hooks(self) -> List[HookMetadata]:
        """Return metadata for all hooks (installed, removed, failed)."""
        with self._lock:
            return list(self._hooks.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._originals)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            installed = sum(1 for h in self._hooks.values() if h.status == HookStatus.INSTALLED)
            failed = sum(1 for h in self._hooks.values() if h.status == HookStatus.FAILED)
            removed = sum(1 for h in self._hooks.values() if h.status == HookStatus.REMOVED)
        return {"installed": installed, "failed": failed, "removed": removed}


# ===================================================================
#  SafeUnpatch context manager
# ===================================================================

@contextmanager
def safe_unpatch(registry: HookRegistry) -> Generator[HookRegistry, None, None]:
    """Ensure all hooks in *registry* are removed, even on exception.

    Usage::

        with safe_unpatch(registry):
            interceptor.install()
            run_pipeline()
    """
    try:
        yield registry
    finally:
        registry.remove_all()


# ===================================================================
#  PandasInterceptor
# ===================================================================

class PandasInterceptor:
    """Patches pandas DataFrame, Series, GroupBy, and module-level functions
    to capture operation metadata for DAG construction.

    Usage::

        interceptor = PandasInterceptor()
        interceptor.install()
        try:
            run_pipeline()
        finally:
            interceptor.uninstall()

    Or with the context manager::

        with PandasInterceptor() as interceptor:
            run_pipeline()
        records = interceptor.records
    """

    def __init__(self, provenance_callback: Optional[Callable[..., Optional[ProvenanceInfo]]] = None) -> None:
        self._registry = HookRegistry()
        self._records: List[PandasInterceptRecord] = []
        self._lock = threading.Lock()
        self._active = False
        self._provenance_callback = provenance_callback

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def registry(self) -> HookRegistry:
        return self._registry

    @property
    def records(self) -> List[PandasInterceptRecord]:
        with self._lock:
            return list(self._records)

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    #  Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PandasInterceptor":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.uninstall()

    # ------------------------------------------------------------------
    #  Install / uninstall
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install all hooks.  Safe to call multiple times."""
        if self._active:
            return
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed – PandasInterceptor has no effect")
            return

        self._install_dataframe_hooks(pd)
        self._install_series_hooks(pd)
        self._install_groupby_hooks(pd)
        self._install_module_hooks(pd)
        self._active = True

    def uninstall(self) -> None:
        """Remove all hooks, restoring original methods."""
        self._registry.remove_all()
        self._active = False

    # ------------------------------------------------------------------
    #  DataFrame hooks
    # ------------------------------------------------------------------

    def _install_dataframe_hooks(self, pd: Any) -> None:
        """Patch methods on ``pd.DataFrame``."""
        df_cls = pd.DataFrame
        for method_name, op_type in _DF_METHOD_MAP.items():
            if not hasattr(df_cls, method_name):
                continue
            original = getattr(df_cls, method_name)
            wrapper = self._make_instance_wrapper(original, method_name, op_type, "pandas.DataFrame")
            meta = HookMetadata(
                target_class="pandas.DataFrame",
                method_name=method_name,
                op_type=op_type,
            )
            self._registry.register(df_cls, method_name, wrapper, meta)

    # ------------------------------------------------------------------
    #  Series hooks
    # ------------------------------------------------------------------

    def _install_series_hooks(self, pd: Any) -> None:
        """Patch methods on ``pd.Series``."""
        series_cls = pd.Series
        for method_name, op_type in _SERIES_METHOD_MAP.items():
            if not hasattr(series_cls, method_name):
                continue
            original = getattr(series_cls, method_name)
            wrapper = self._make_instance_wrapper(original, method_name, op_type, "pandas.Series")
            meta = HookMetadata(
                target_class="pandas.Series",
                method_name=method_name,
                op_type=op_type,
            )
            self._registry.register(series_cls, method_name, wrapper, meta)

    # ------------------------------------------------------------------
    #  GroupBy hooks
    # ------------------------------------------------------------------

    def _install_groupby_hooks(self, pd: Any) -> None:
        """Patch methods on ``pd.core.groupby.GroupBy``."""
        groupby_classes: List[Any] = []
        try:
            groupby_classes.append(pd.core.groupby.DataFrameGroupBy)
        except AttributeError:
            pass
        try:
            groupby_classes.append(pd.core.groupby.SeriesGroupBy)
        except AttributeError:
            pass

        for gb_cls in groupby_classes:
            cls_name = _safe_class_name(gb_cls)
            for method_name, op_type in _GROUPBY_METHOD_MAP.items():
                if not hasattr(gb_cls, method_name):
                    continue
                original = getattr(gb_cls, method_name)
                wrapper = self._make_instance_wrapper(original, method_name, op_type, cls_name)
                meta = HookMetadata(
                    target_class=cls_name,
                    method_name=method_name,
                    op_type=op_type,
                )
                self._registry.register(gb_cls, method_name, wrapper, meta)

    # ------------------------------------------------------------------
    #  Module-level function hooks
    # ------------------------------------------------------------------

    def _install_module_hooks(self, pd: Any) -> None:
        """Patch module-level functions (``pd.concat``, ``pd.read_csv``, etc.)."""
        for func_name, op_type in _MODULE_FUNC_MAP.items():
            if not hasattr(pd, func_name):
                continue
            original = getattr(pd, func_name)
            wrapper = self._make_module_wrapper(original, func_name, op_type)
            meta = HookMetadata(
                target_class="pandas",
                method_name=func_name,
                op_type=op_type,
            )
            self._registry.register(pd, func_name, wrapper, meta)

    # ------------------------------------------------------------------
    #  Wrapper factories
    # ------------------------------------------------------------------

    def _make_instance_wrapper(
        self,
        original: Any,
        method_name: str,
        op_type: OpType,
        class_name: str,
    ) -> Callable[..., Any]:
        """Create a wrapper for a bound instance method (DataFrame/Series/GroupBy)."""
        interceptor = self

        @functools.wraps(original)
        def wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
            if not interceptor._active:
                return original(self_obj, *args, **kwargs)

            input_shape = _safe_shape(self_obj)
            input_cols = _safe_columns(self_obj)

            extra: Dict[str, Any] = {}
            if method_name in ("merge", "join"):
                extra = _extract_merge_extra(kwargs, args)
            elif method_name == "groupby":
                extra = _extract_groupby_extra(kwargs, args)

            t0 = time.monotonic()
            try:
                result = original(self_obj, *args, **kwargs)
            except Exception:
                interceptor._bump_error(class_name, method_name)
                raise

            wall_ms = (time.monotonic() - t0) * 1000.0

            output_shape = _safe_shape(result)
            output_cols = _safe_columns(result)

            prov: Optional[ProvenanceInfo] = None
            if interceptor._provenance_callback is not None:
                try:
                    prov = interceptor._provenance_callback(
                        self_obj, result, method_name, op_type,
                    )
                except Exception:
                    pass

            record = PandasInterceptRecord(
                timestamp=time.monotonic(),
                op_type=op_type,
                method_name=method_name,
                target_class=class_name,
                input_shape=input_shape,
                output_shape=output_shape,
                input_columns=input_cols,
                output_columns=output_cols,
                kwargs_keys=list(kwargs.keys()),
                wall_time_ms=wall_ms,
                provenance=prov,
                extra=extra,
            )
            interceptor._record(record, class_name, method_name)
            return result

        return wrapper

    def _make_module_wrapper(
        self,
        original: Any,
        func_name: str,
        op_type: OpType,
    ) -> Callable[..., Any]:
        """Create a wrapper for a module-level function (``pd.concat``, etc.)."""
        interceptor = self

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not interceptor._active:
                return original(*args, **kwargs)

            extra: Dict[str, Any] = {}
            if func_name == "concat":
                extra = _extract_concat_extra(args, kwargs)
            elif func_name.startswith("read_"):
                extra = _extract_read_extra(func_name, args, kwargs)
            elif func_name == "merge":
                extra = _extract_merge_extra(kwargs, args)

            t0 = time.monotonic()
            try:
                result = original(*args, **kwargs)
            except Exception:
                interceptor._bump_error("pandas", func_name)
                raise

            wall_ms = (time.monotonic() - t0) * 1000.0

            output_shape = _safe_shape(result)
            output_cols = _safe_columns(result)

            prov: Optional[ProvenanceInfo] = None
            if interceptor._provenance_callback is not None:
                try:
                    prov = interceptor._provenance_callback(
                        None, result, func_name, op_type,
                    )
                except Exception:
                    pass

            record = PandasInterceptRecord(
                timestamp=time.monotonic(),
                op_type=op_type,
                method_name=func_name,
                target_class="pandas",
                input_shape=None,
                output_shape=output_shape,
                input_columns=[],
                output_columns=output_cols,
                kwargs_keys=list(kwargs.keys()),
                wall_time_ms=wall_ms,
                provenance=prov,
                extra=extra,
            )
            interceptor._record(record, "pandas", func_name)
            return result

        return wrapper

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _record(self, record: PandasInterceptRecord, class_name: str, method_name: str) -> None:
        """Thread-safe recording of an intercept record."""
        with self._lock:
            self._records.append(record)
        key = f"{class_name}.{method_name}"
        with self._registry._lock:
            meta = self._registry._hooks.get(key)
            if meta is not None:
                meta.call_count += 1

    def _bump_error(self, class_name: str, method_name: str) -> None:
        """Increment the error counter for a hook."""
        key = f"{class_name}.{method_name}"
        with self._registry._lock:
            meta = self._registry._hooks.get(key)
            if meta is not None:
                meta.error_count += 1

    # ------------------------------------------------------------------
    #  Query helpers
    # ------------------------------------------------------------------

    def records_by_op(self, op_type: OpType) -> List[PandasInterceptRecord]:
        """Return records filtered by operation type."""
        with self._lock:
            return [r for r in self._records if r.op_type == op_type]

    def records_by_method(self, method_name: str) -> List[PandasInterceptRecord]:
        """Return records filtered by method name."""
        with self._lock:
            return [r for r in self._records if r.method_name == method_name]

    def records_by_class(self, class_substr: str) -> List[PandasInterceptRecord]:
        """Return records whose target_class contains *class_substr*."""
        with self._lock:
            return [r for r in self._records if class_substr in r.target_class]

    def io_records(self) -> List[PandasInterceptRecord]:
        """Return only data-source (IO) records."""
        io_ops = {
            OpType.READ_CSV, OpType.READ_PARQUET, OpType.READ_JSON,
            OpType.READ_EXCEL, OpType.READ_SQL, OpType.READ_HDF,
            OpType.READ_FEATHER,
        }
        with self._lock:
            return [r for r in self._records if r.op_type in io_ops]

    def shape_changing_records(self) -> List[PandasInterceptRecord]:
        """Return records where the output shape differs from input shape."""
        with self._lock:
            return [
                r for r in self._records
                if r.input_shape is not None
                and r.output_shape is not None
                and r.input_shape != r.output_shape
            ]

    def column_changing_records(self) -> List[PandasInterceptRecord]:
        """Return records where output columns differ from input columns."""
        with self._lock:
            return [
                r for r in self._records
                if r.input_columns and r.output_columns
                and set(r.input_columns) != set(r.output_columns)
            ]

    def dag_nodes(self) -> List[DAGNode]:
        """Convert intercept records to preliminary :class:`DAGNode` instances."""
        nodes: List[DAGNode] = []
        for record in self.records:
            shape: Optional[ShapeMetadata] = None
            if record.output_shape is not None:
                n_test = 0
                if record.provenance is not None:
                    n_test = int(record.output_shape[0] * record.provenance.test_fraction)
                shape = ShapeMetadata(
                    n_rows=record.output_shape[0],
                    n_cols=record.output_shape[1],
                    n_test_rows=n_test,
                )
            node = NodeFactory.create(
                op_type=record.op_type,
                shape=shape,
                provenance=record.provenance,
            )
            node.annotations["pandas_record_id"] = record.record_id
            node.annotations["wall_time_ms"] = record.wall_time_ms
            if record.extra:
                node.annotations["extra"] = record.extra
            nodes.append(node)
        return nodes

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary of interception activity."""
        with self._lock:
            n_records = len(self._records)
            ops_dist: Dict[str, int] = {}
            for r in self._records:
                ops_dist[r.op_type.value] = ops_dist.get(r.op_type.value, 0) + 1
        return {
            "active": self._active,
            "n_records": n_records,
            "n_hooks": len(self._registry),
            "hook_summary": self._registry.summary(),
            "operation_distribution": ops_dist,
        }

    def clear_records(self) -> None:
        """Discard all recorded intercept records."""
        with self._lock:
            self._records.clear()

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Export all records as dictionaries."""
        with self._lock:
            return [r.to_dict() for r in self._records]
