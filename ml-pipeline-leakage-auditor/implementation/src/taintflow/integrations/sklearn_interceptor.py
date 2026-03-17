"""
taintflow.integrations.sklearn_interceptor – Scikit-learn integration.

Provides audited wrappers for scikit-learn estimators, transformers, and
pipelines.  Each wrapper preserves the original sklearn API while recording
input/output shapes, train/test provenance, and operation metadata to a
shared :class:`~taintflow.dag.builder.DAGBuilder`.

Usage::

    from taintflow.integrations.sklearn_interceptor import (
        AuditedPipeline, AuditedStandardScaler,
    )
    pipe = AuditedPipeline([("scaler", AuditedStandardScaler())])
    pipe.fit(X_train, y_train)
    # All operations are automatically recorded for leakage analysis.
"""

from __future__ import annotations

import inspect
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
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

try:
    import sklearn
    import sklearn.base
    import sklearn.pipeline

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ===================================================================
#  Version detection
# ===================================================================


class SklearnVersionDetector:
    """Detect the installed scikit-learn version and feature availability.

    This is used to adapt behaviour for different sklearn releases (e.g.
    ``set_output``, ``feature_names_out_``, ``TargetEncoder``).
    """

    def __init__(self) -> None:
        self._version: Optional[str] = None
        self._major: int = 0
        self._minor: int = 0
        self._patch: int = 0
        self._detect()

    def _detect(self) -> None:
        """Parse the sklearn version string."""
        if not _SKLEARN_AVAILABLE:
            return
        self._version = sklearn.__version__
        parts = self._version.split(".")
        try:
            self._major = int(parts[0])
            self._minor = int(parts[1]) if len(parts) > 1 else 0
            self._patch = int(parts[2].split("rc")[0].split("dev")[0]) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            pass

    @property
    def version(self) -> Optional[str]:
        """Return the detected sklearn version string, or *None*."""
        return self._version

    @property
    def version_tuple(self) -> Tuple[int, int, int]:
        """Return ``(major, minor, patch)``."""
        return (self._major, self._minor, self._patch)

    @property
    def is_available(self) -> bool:
        """Return *True* if sklearn is importable."""
        return _SKLEARN_AVAILABLE

    def has_set_output(self) -> bool:
        """Return *True* if ``set_output`` API is available (>=1.2)."""
        return self._major >= 1 and self._minor >= 2

    def has_feature_names_out(self) -> bool:
        """Return *True* if ``get_feature_names_out`` is available (>=1.0)."""
        return self._major >= 1

    def has_target_encoder(self) -> bool:
        """Return *True* if ``TargetEncoder`` is available (>=1.3)."""
        return self._major >= 1 and self._minor >= 3

    def __repr__(self) -> str:
        return f"SklearnVersionDetector(version={self._version!r})"


_VERSION_DETECTOR = SklearnVersionDetector()


# ===================================================================
#  Estimator classification
# ===================================================================


@dataclass
class _EstimatorProfile:
    """Internal profile describing an estimator's leakage-relevant properties."""

    class_name: str
    is_transformer: bool = False
    is_classifier: bool = False
    is_regressor: bool = False
    is_clusterer: bool = False
    uses_target: bool = False
    is_stateful: bool = False
    leaks_on_fit: bool = False
    op_type: OpType = OpType.UNKNOWN
    description: str = ""


class EstimatorClassifier:
    """Classify any sklearn estimator by its leakage-relevant properties.

    Given an estimator instance (or class), determine whether it is a
    transformer, classifier, regressor, or clusterer, and whether its
    ``fit`` method is stateful in a way that can channel test information
    into training.
    """

    _OP_TYPE_MAP: Dict[str, OpType] = {
        "StandardScaler": OpType.STANDARD_SCALER,
        "MinMaxScaler": OpType.MINMAX_SCALER,
        "RobustScaler": OpType.ROBUST_SCALER,
        "Normalizer": OpType.NORMALIZER,
        "PCA": OpType.STANDARD_SCALER,
        "TruncatedSVD": OpType.STANDARD_SCALER,
        "SimpleImputer": OpType.IMPUTER,
        "KNNImputer": OpType.KNN_IMPUTER,
        "OneHotEncoder": OpType.ONEHOT_ENCODER,
        "OrdinalEncoder": OpType.ORDINAL_ENCODER,
        "TargetEncoder": OpType.TARGET_ENCODER,
        "LabelEncoder": OpType.LABEL_ENCODER,
        "SelectKBest": OpType.FIT,
        "VarianceThreshold": OpType.FIT,
        "PolynomialFeatures": OpType.POLYNOMIAL_FEATURES,
        "KBinsDiscretizer": OpType.KBINS_DISCRETIZER,
        "Binarizer": OpType.BINARIZER,
    }

    _LEAKS_ON_FIT: Set[str] = {
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "PCA",
        "TruncatedSVD",
        "SimpleImputer",
        "KNNImputer",
        "TargetEncoder",
        "SelectKBest",
        "VarianceThreshold",
    }

    @classmethod
    def classify(cls, estimator: Any) -> _EstimatorProfile:
        """Return a :class:`_EstimatorProfile` for *estimator*.

        Parameters
        ----------
        estimator
            An sklearn estimator instance or class.

        Returns
        -------
        _EstimatorProfile
            Leakage-relevant properties of the estimator.
        """
        est_class: type = estimator if isinstance(estimator, type) else type(estimator)
        class_name = est_class.__name__

        is_transformer = False
        is_classifier = False
        is_regressor = False
        is_clusterer = False
        uses_target = False

        if _SKLEARN_AVAILABLE:
            is_transformer = hasattr(estimator, "transform") and hasattr(estimator, "fit")
            is_classifier = getattr(estimator, "_estimator_type", None) == "classifier"
            is_regressor = getattr(estimator, "_estimator_type", None) == "regressor"
            is_clusterer = getattr(estimator, "_estimator_type", None) == "clusterer"

            fit_sig = inspect.signature(est_class.fit) if hasattr(est_class, "fit") else None
            if fit_sig is not None:
                params = list(fit_sig.parameters.keys())
                uses_target = "y" in params and params.index("y") <= 2
        else:
            is_transformer = hasattr(estimator, "transform")
            uses_target = hasattr(estimator, "fit")

        op_type = cls._OP_TYPE_MAP.get(class_name, OpType.FIT)
        leaks = class_name in cls._LEAKS_ON_FIT

        return _EstimatorProfile(
            class_name=class_name,
            is_transformer=is_transformer,
            is_classifier=is_classifier,
            is_regressor=is_regressor,
            is_clusterer=is_clusterer,
            uses_target=uses_target,
            is_stateful=leaks,
            leaks_on_fit=leaks,
            op_type=op_type,
            description=f"{class_name}(transformer={is_transformer}, target={uses_target})",
        )

    @classmethod
    def may_leak(cls, estimator: Any) -> bool:
        """Return *True* if *estimator* may channel test data into training."""
        return cls.classify(estimator).leaks_on_fit

    @classmethod
    def op_type_for(cls, estimator: Any) -> OpType:
        """Return the :class:`OpType` for *estimator*."""
        return cls.classify(estimator).op_type


# ===================================================================
#  Shape / provenance helpers
# ===================================================================


def _extract_shape(X: Any) -> Tuple[int, int]:
    """Best-effort extraction of ``(n_rows, n_cols)`` from array-like data."""
    if hasattr(X, "shape"):
        shape = X.shape
        n_rows = int(shape[0]) if len(shape) >= 1 else 0
        n_cols = int(shape[1]) if len(shape) >= 2 else 1
        return (n_rows, n_cols)
    if hasattr(X, "__len__"):
        n_rows = len(X)
        first = X[0] if n_rows > 0 else None
        n_cols = len(first) if first is not None and hasattr(first, "__len__") else 1
        return (n_rows, n_cols)
    return (0, 0)


def _make_shape_meta(
    X: Any,
    *,
    n_test_rows: int = 0,
    n_train_rows: int = 0,
) -> ShapeMetadata:
    """Build :class:`ShapeMetadata` from array-like *X*."""
    n_rows, n_cols = _extract_shape(X)
    return ShapeMetadata(
        n_rows=n_rows,
        n_cols=n_cols,
        n_test_rows=n_test_rows,
        n_train_rows=n_train_rows,
    )


def _caller_location(depth: int = 2) -> SourceLocation:
    """Capture the call-site source location *depth* frames up the stack."""
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            info = inspect.getframeinfo(frame)
            return SourceLocation(
                file_path=info.filename,
                line_number=info.lineno,
                function_name=info.function,
            )
    finally:
        del frame
    return SourceLocation(file_path="<unknown>", line_number=0)


def _generate_node_id(prefix: str = "sk") -> str:
    """Generate a unique node identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ===================================================================
#  Shared audit log
# ===================================================================


@dataclass
class _AuditRecord:
    """A single record in the audit trail."""

    node_id: str
    op_type: OpType
    estimator_class: str
    method: str
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    origin: Optional[Origin] = None
    source_location: Optional[SourceLocation] = None
    wall_time_ms: float = 0.0
    provenance: Optional[ProvenanceInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class _AuditLog:
    """Thread-local log of audit records, shared between wrappers."""

    def __init__(self) -> None:
        self._records: List[_AuditRecord] = []
        self._dag_nodes: List[DAGNode] = []

    def record(self, rec: _AuditRecord) -> None:
        """Append an audit record and create the corresponding DAG node."""
        self._records.append(rec)
        node = NodeFactory.create(
            op_type=rec.op_type,
            source_location=rec.source_location,
            shape=ShapeMetadata(
                n_rows=rec.output_shape[0],
                n_cols=rec.output_shape[1],
            ),
            provenance=rec.provenance,
        )
        self._dag_nodes.append(node)
        logger.debug(
            "Audit: %s.%s  in=%s out=%s  %.1f ms",
            rec.estimator_class,
            rec.method,
            rec.input_shape,
            rec.output_shape,
            rec.wall_time_ms,
        )

    @property
    def records(self) -> List[_AuditRecord]:
        """Return all collected records."""
        return list(self._records)

    @property
    def dag_nodes(self) -> List[DAGNode]:
        """Return all DAG nodes created from records."""
        return list(self._dag_nodes)

    def clear(self) -> None:
        """Discard all records."""
        self._records.clear()
        self._dag_nodes.clear()

    def __len__(self) -> int:
        return len(self._records)


# Module-level default audit log
_default_audit_log = _AuditLog()


def get_audit_log() -> _AuditLog:
    """Return the module-level audit log."""
    return _default_audit_log


def reset_audit_log() -> None:
    """Clear the module-level audit log."""
    _default_audit_log.clear()


# ===================================================================
#  AuditedTransformerMixin
# ===================================================================


class AuditedTransformerMixin:
    """Mixin that intercepts ``fit``, ``transform``, and ``fit_transform``.

    Subclass this together with the target sklearn estimator to automatically
    record every call to the audit log.  The mixin delegates to the original
    methods so the sklearn API is fully preserved.

    Attributes:
        _audit_log: The :class:`_AuditLog` used for recording.
        _audit_origin: The partition origin for the data being processed.
    """

    _audit_log: _AuditLog
    _audit_origin: Optional[Origin]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def _init_audit(
        self,
        audit_log: Optional[_AuditLog] = None,
        origin: Optional[Origin] = None,
    ) -> None:
        """Initialise auditing state (called by subclass ``__init__``)."""
        self._audit_log = audit_log or _default_audit_log
        self._audit_origin = origin

    def _record_operation(
        self,
        method: str,
        X_in: Any,
        result: Any,
        wall_time_ms: float,
        op_type: Optional[OpType] = None,
    ) -> None:
        """Record a single fit/transform/predict call."""
        est_class = type(self).__name__
        in_shape = _extract_shape(X_in)
        out_shape = _extract_shape(result) if result is not None else in_shape

        if op_type is None:
            op_type = {
                "fit": OpType.FIT,
                "transform": OpType.TRANSFORM_SK,
                "fit_transform": OpType.FIT_TRANSFORM,
                "predict": OpType.PREDICT,
                "predict_proba": OpType.PREDICT_PROBA,
                "score": OpType.SCORE,
                "inverse_transform": OpType.INVERSE_TRANSFORM,
            }.get(method, OpType.UNKNOWN)

        provenance = None
        if self._audit_origin is not None:
            test_frac = 1.0 if self._audit_origin == Origin.TEST else 0.0
            provenance = ProvenanceInfo(
                test_fraction=test_frac,
                origin_set=frozenset({self._audit_origin}),
            )

        rec = _AuditRecord(
            node_id=_generate_node_id(est_class.lower()),
            op_type=op_type,
            estimator_class=est_class,
            method=method,
            input_shape=in_shape,
            output_shape=out_shape,
            origin=self._audit_origin,
            source_location=_caller_location(depth=3),
            wall_time_ms=wall_time_ms,
            provenance=provenance,
        )
        self._audit_log.record(rec)

    # -- intercepted methods --------------------------------------------------

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit the estimator, recording the operation."""
        t0 = time.perf_counter()
        result = super().fit(X, y, **fit_params)  # type: ignore[misc]
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_operation("fit", X, X, elapsed, OpType.FIT)
        return result

    def transform(self, X: Any, **params: Any) -> Any:
        """Transform data, recording the operation."""
        t0 = time.perf_counter()
        result = super().transform(X, **params)  # type: ignore[misc]
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_operation("transform", X, result, elapsed, OpType.TRANSFORM_SK)
        return result

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit and transform, recording as a single operation."""
        t0 = time.perf_counter()
        if hasattr(super(), "fit_transform"):
            result = super().fit_transform(X, y, **fit_params)  # type: ignore[misc]
        else:
            super().fit(X, y, **fit_params)  # type: ignore[misc]
            result = super().transform(X)  # type: ignore[misc]
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_operation("fit_transform", X, result, elapsed, OpType.FIT_TRANSFORM)
        return result


# ===================================================================
#  AuditedEstimator – generic wrapper
# ===================================================================


class AuditedEstimator:
    """Generic wrapper that adds auditing to any sklearn estimator.

    This is a *composition*-based wrapper: it holds a reference to the
    original estimator and delegates all attribute access to it, while
    intercepting ``fit``, ``transform``, ``predict``, and related methods.

    Parameters
    ----------
    estimator
        The sklearn estimator instance to wrap.
    audit_log
        Optional shared audit log; defaults to the module-level log.
    origin
        Optional data-partition origin for provenance tracking.
    """

    def __init__(
        self,
        estimator: Any,
        audit_log: Optional[_AuditLog] = None,
        origin: Optional[Origin] = None,
    ) -> None:
        self._estimator = estimator
        self._audit_log = audit_log or _default_audit_log
        self._audit_origin = origin
        self._profile = EstimatorClassifier.classify(estimator)

    # -- delegation -----------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped estimator."""
        return getattr(self._estimator, name)

    def __repr__(self) -> str:
        return f"AuditedEstimator({self._estimator!r})"

    @property
    def estimator(self) -> Any:
        """Return the wrapped estimator."""
        return self._estimator

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return estimator parameters (sklearn API)."""
        return self._estimator.get_params(deep=deep)

    def set_params(self, **params: Any) -> "AuditedEstimator":
        """Set estimator parameters (sklearn API)."""
        self._estimator.set_params(**params)
        return self

    # -- audited methods ------------------------------------------------------

    def _record(
        self,
        method: str,
        X_in: Any,
        result: Any,
        wall_ms: float,
        op_type: OpType,
    ) -> None:
        est_class = type(self._estimator).__name__
        in_shape = _extract_shape(X_in)
        out_shape = _extract_shape(result) if result is not None else in_shape

        provenance = None
        if self._audit_origin is not None:
            test_frac = 1.0 if self._audit_origin == Origin.TEST else 0.0
            provenance = ProvenanceInfo(
                test_fraction=test_frac,
                origin_set=frozenset({self._audit_origin}),
            )

        rec = _AuditRecord(
            node_id=_generate_node_id(est_class.lower()),
            op_type=op_type,
            estimator_class=est_class,
            method=method,
            input_shape=in_shape,
            output_shape=out_shape,
            origin=self._audit_origin,
            source_location=_caller_location(depth=3),
            wall_time_ms=wall_ms,
            provenance=provenance,
        )
        self._audit_log.record(rec)

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "AuditedEstimator":
        """Fit the wrapped estimator, recording the operation."""
        t0 = time.perf_counter()
        self._estimator.fit(X, y, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("fit", X, X, elapsed, OpType.FIT)
        return self

    def transform(self, X: Any, **kwargs: Any) -> Any:
        """Transform *X* using the wrapped estimator, recording the operation."""
        t0 = time.perf_counter()
        result = self._estimator.transform(X, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("transform", X, result, elapsed, OpType.TRANSFORM_SK)
        return result

    def fit_transform(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
        """Fit and transform, recorded as a single operation."""
        t0 = time.perf_counter()
        if hasattr(self._estimator, "fit_transform"):
            result = self._estimator.fit_transform(X, y, **kwargs)
        else:
            self._estimator.fit(X, y, **kwargs)
            result = self._estimator.transform(X)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("fit_transform", X, result, elapsed, OpType.FIT_TRANSFORM)
        return result

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Predict using the wrapped estimator, recording the operation."""
        t0 = time.perf_counter()
        result = self._estimator.predict(X, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("predict", X, result, elapsed, OpType.PREDICT)
        return result

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        """Predict probabilities, recording the operation."""
        t0 = time.perf_counter()
        result = self._estimator.predict_proba(X, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("predict_proba", X, result, elapsed, OpType.PREDICT_PROBA)
        return result

    def score(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
        """Score the estimator, recording the operation."""
        t0 = time.perf_counter()
        result = self._estimator.score(X, y, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("score", X, result, elapsed, OpType.SCORE)
        return result

    def inverse_transform(self, X: Any, **kwargs: Any) -> Any:
        """Inverse-transform, recording the operation."""
        t0 = time.perf_counter()
        result = self._estimator.inverse_transform(X, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record("inverse_transform", X, result, elapsed, OpType.INVERSE_TRANSFORM)
        return result


# ===================================================================
#  Concrete audited wrappers  (mixin + sklearn class)
# ===================================================================

def _make_audited_wrapper(
    sklearn_import_path: str,
    class_name: str,
    op_type: OpType,
) -> Type:
    """Dynamically build an audited wrapper class.

    If sklearn is not available the wrapper is still created but will raise
    ``ImportError`` on instantiation.

    Parameters
    ----------
    sklearn_import_path
        Dotted path to the sklearn module containing the base class.
    class_name
        Name of the sklearn class to wrap.
    op_type
        The default :class:`OpType` to record on ``fit``.

    Returns
    -------
    type
        A new class that inherits from :class:`AuditedTransformerMixin` and
        the sklearn base class.
    """
    base_cls: Optional[type] = None

    if _SKLEARN_AVAILABLE:
        try:
            import importlib

            mod = importlib.import_module(sklearn_import_path)
            base_cls = getattr(mod, class_name, None)
        except (ImportError, AttributeError):
            pass

    if base_cls is not None:
        # MRO: AuditedTransformerMixin → sklearn class → object
        wrapper_cls = type(
            f"Audited{class_name}",
            (AuditedTransformerMixin, base_cls),
            {
                "__doc__": (
                    f"Audited wrapper around ``sklearn.{sklearn_import_path}.{class_name}``.\n\n"
                    f"Preserves the full sklearn API while recording every ``fit``, "
                    f"``transform``, and ``fit_transform`` call to the TaintFlow "
                    f"audit log with shapes and provenance metadata.\n"
                ),
                "_default_op_type": op_type,
            },
        )

        original_init = base_cls.__init__

        def _audited_init(
            self: Any,
            *args: Any,
            audit_log: Optional[_AuditLog] = None,
            origin: Optional[Origin] = None,
            **kwargs: Any,
        ) -> None:
            original_init(self, *args, **kwargs)
            self._init_audit(audit_log=audit_log, origin=origin)

        wrapper_cls.__init__ = _audited_init  # type: ignore[attr-defined]
        return wrapper_cls

    # sklearn not installed – build a stub that raises on instantiation
    class _StubWrapper(AuditedTransformerMixin):
        __doc__ = (
            f"Stub for ``Audited{class_name}`` (sklearn not installed).\n\n"
            f"Install scikit-learn to use this wrapper."
        )
        _default_op_type = op_type

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                f"scikit-learn is required for Audited{class_name}. "
                f"Install it with: pip install scikit-learn"
            )

    _StubWrapper.__name__ = f"Audited{class_name}"
    _StubWrapper.__qualname__ = f"Audited{class_name}"
    return _StubWrapper


# -- Scalers ---------------------------------------------------------------

AuditedStandardScaler: Type = _make_audited_wrapper(
    "sklearn.preprocessing", "StandardScaler", OpType.STANDARD_SCALER,
)

AuditedMinMaxScaler: Type = _make_audited_wrapper(
    "sklearn.preprocessing", "MinMaxScaler", OpType.MINMAX_SCALER,
)

AuditedRobustScaler: Type = _make_audited_wrapper(
    "sklearn.preprocessing", "RobustScaler", OpType.ROBUST_SCALER,
)

# -- Decomposition ---------------------------------------------------------

AuditedPCA: Type = _make_audited_wrapper(
    "sklearn.decomposition", "PCA", OpType.STANDARD_SCALER,
)

AuditedTruncatedSVD: Type = _make_audited_wrapper(
    "sklearn.decomposition", "TruncatedSVD", OpType.STANDARD_SCALER,
)

# -- Imputation -------------------------------------------------------------

AuditedSimpleImputer: Type = _make_audited_wrapper(
    "sklearn.impute", "SimpleImputer", OpType.IMPUTER,
)

AuditedKNNImputer: Type = _make_audited_wrapper(
    "sklearn.impute", "KNNImputer", OpType.KNN_IMPUTER,
)

# -- Encoding ---------------------------------------------------------------

AuditedOneHotEncoder: Type = _make_audited_wrapper(
    "sklearn.preprocessing", "OneHotEncoder", OpType.ONEHOT_ENCODER,
)

AuditedOrdinalEncoder: Type = _make_audited_wrapper(
    "sklearn.preprocessing", "OrdinalEncoder", OpType.ORDINAL_ENCODER,
)

AuditedTargetEncoder: Type = _make_audited_wrapper(
    "sklearn.preprocessing", "TargetEncoder", OpType.TARGET_ENCODER,
)

# -- Feature selection -------------------------------------------------------

AuditedSelectKBest: Type = _make_audited_wrapper(
    "sklearn.feature_selection", "SelectKBest", OpType.FIT,
)

AuditedVarianceThreshold: Type = _make_audited_wrapper(
    "sklearn.feature_selection", "VarianceThreshold", OpType.FIT,
)


# ===================================================================
#  AuditedPipeline
# ===================================================================


class AuditedPipeline:
    """Wraps an sklearn :class:`~sklearn.pipeline.Pipeline` with automatic
    leakage auditing.

    Every ``fit`` / ``transform`` / ``predict`` call on the pipeline is
    intercepted so that each stage's input and output shapes, provenance,
    and timings are recorded to the shared audit log.

    Parameters
    ----------
    steps
        A list of ``(name, estimator)`` tuples, exactly like
        ``sklearn.pipeline.Pipeline``.
    memory
        Optional caching directory (passed through to sklearn).
    verbose
        Verbosity flag (passed through to sklearn).
    audit_log
        Optional shared audit log.
    origin
        Optional data-partition origin.
    """

    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        *,
        memory: Any = None,
        verbose: bool = False,
        audit_log: Optional[_AuditLog] = None,
        origin: Optional[Origin] = None,
    ) -> None:
        self._audit_log = audit_log or _default_audit_log
        self._audit_origin = origin
        self._steps = steps
        self._step_names = [name for name, _ in steps]

        if _SKLEARN_AVAILABLE:
            self._pipeline = sklearn.pipeline.Pipeline(
                steps=steps, memory=memory, verbose=verbose,
            )
        else:
            self._pipeline = None

    def __getattr__(self, name: str) -> Any:
        if self._pipeline is not None:
            return getattr(self._pipeline, name)
        raise AttributeError(f"AuditedPipeline has no attribute {name!r} (sklearn not installed)")

    def __repr__(self) -> str:
        return f"AuditedPipeline(steps={self._step_names})"

    @property
    def steps(self) -> List[Tuple[str, Any]]:
        """Return the pipeline steps."""
        if self._pipeline is not None:
            return self._pipeline.steps
        return list(self._steps)

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Return a dict mapping step names to estimators."""
        if self._pipeline is not None:
            return dict(self._pipeline.named_steps)
        return {name: est for name, est in self._steps}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return pipeline parameters (sklearn API)."""
        if self._pipeline is not None:
            return self._pipeline.get_params(deep=deep)
        return {"steps": self._steps}

    def set_params(self, **kwargs: Any) -> "AuditedPipeline":
        """Set pipeline parameters (sklearn API)."""
        if self._pipeline is not None:
            self._pipeline.set_params(**kwargs)
        return self

    def _record_step(
        self,
        step_name: str,
        estimator: Any,
        method: str,
        X_in: Any,
        X_out: Any,
        wall_ms: float,
    ) -> None:
        """Record a single pipeline step execution."""
        est_class = type(estimator).__name__
        op_type = EstimatorClassifier.op_type_for(estimator)
        if method == "transform":
            op_type = OpType.TRANSFORM_SK
        elif method == "predict":
            op_type = OpType.PREDICT
        elif method == "fit_transform":
            op_type = OpType.FIT_TRANSFORM

        in_shape = _extract_shape(X_in)
        out_shape = _extract_shape(X_out) if X_out is not None else in_shape

        provenance = None
        if self._audit_origin is not None:
            test_frac = 1.0 if self._audit_origin == Origin.TEST else 0.0
            provenance = ProvenanceInfo(
                test_fraction=test_frac,
                origin_set=frozenset({self._audit_origin}),
            )

        rec = _AuditRecord(
            node_id=_generate_node_id(f"pipe_{step_name}"),
            op_type=op_type,
            estimator_class=est_class,
            method=method,
            input_shape=in_shape,
            output_shape=out_shape,
            origin=self._audit_origin,
            source_location=_caller_location(depth=4),
            wall_time_ms=wall_ms,
            provenance=provenance,
            metadata={"pipeline_step": step_name},
        )
        self._audit_log.record(rec)

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "AuditedPipeline":
        """Fit all steps, recording each stage individually."""
        if self._pipeline is None:
            raise ImportError("scikit-learn is required for AuditedPipeline.fit")

        Xt = X
        for step_name, estimator in self._pipeline.steps[:-1]:
            t0 = time.perf_counter()
            if hasattr(estimator, "fit_transform"):
                Xt_new = estimator.fit_transform(Xt, y, **fit_params)
            else:
                estimator.fit(Xt, y, **fit_params)
                Xt_new = estimator.transform(Xt)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record_step(step_name, estimator, "fit_transform", Xt, Xt_new, elapsed)
            Xt = Xt_new

        # Final step – fit only (may be a classifier)
        final_name, final_estimator = self._pipeline.steps[-1]
        t0 = time.perf_counter()
        final_estimator.fit(Xt, y, **fit_params)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_step(final_name, final_estimator, "fit", Xt, Xt, elapsed)
        return self

    def transform(self, X: Any, **params: Any) -> Any:
        """Transform *X* through all transformer steps, recording each."""
        if self._pipeline is None:
            raise ImportError("scikit-learn is required for AuditedPipeline.transform")

        Xt = X
        for step_name, estimator in self._pipeline.steps:
            if not hasattr(estimator, "transform"):
                continue
            t0 = time.perf_counter()
            Xt_new = estimator.transform(Xt, **params)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record_step(step_name, estimator, "transform", Xt, Xt_new, elapsed)
            Xt = Xt_new
        return Xt

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit and transform through all steps, recording each."""
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def predict(self, X: Any, **params: Any) -> Any:
        """Transform through all but the last step, then predict."""
        if self._pipeline is None:
            raise ImportError("scikit-learn is required for AuditedPipeline.predict")

        Xt = X
        for step_name, estimator in self._pipeline.steps[:-1]:
            if not hasattr(estimator, "transform"):
                continue
            t0 = time.perf_counter()
            Xt_new = estimator.transform(Xt, **params)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record_step(step_name, estimator, "transform", Xt, Xt_new, elapsed)
            Xt = Xt_new

        final_name, final_estimator = self._pipeline.steps[-1]
        t0 = time.perf_counter()
        result = final_estimator.predict(Xt, **params)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_step(final_name, final_estimator, "predict", Xt, result, elapsed)
        return result

    def predict_proba(self, X: Any, **params: Any) -> Any:
        """Transform then predict probabilities, recording each step."""
        if self._pipeline is None:
            raise ImportError("scikit-learn is required for AuditedPipeline.predict_proba")

        Xt = X
        for step_name, estimator in self._pipeline.steps[:-1]:
            if not hasattr(estimator, "transform"):
                continue
            t0 = time.perf_counter()
            Xt_new = estimator.transform(Xt, **params)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record_step(step_name, estimator, "transform", Xt, Xt_new, elapsed)
            Xt = Xt_new

        final_name, final_estimator = self._pipeline.steps[-1]
        t0 = time.perf_counter()
        result = final_estimator.predict_proba(Xt, **params)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_step(final_name, final_estimator, "predict_proba", Xt, result, elapsed)
        return result

    def score(self, X: Any, y: Any = None, **params: Any) -> float:
        """Transform then score, recording each step."""
        if self._pipeline is None:
            raise ImportError("scikit-learn is required for AuditedPipeline.score")

        Xt = X
        for step_name, estimator in self._pipeline.steps[:-1]:
            if not hasattr(estimator, "transform"):
                continue
            t0 = time.perf_counter()
            Xt_new = estimator.transform(Xt, **params)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._record_step(step_name, estimator, "transform", Xt, Xt_new, elapsed)
            Xt = Xt_new

        final_name, final_estimator = self._pipeline.steps[-1]
        t0 = time.perf_counter()
        result = final_estimator.score(Xt, y, **params)
        elapsed = (time.perf_counter() - t0) * 1000.0
        self._record_step(final_name, final_estimator, "score", Xt, None, elapsed)
        return result  # type: ignore[return-value]


# ===================================================================
#  PipelineAuditor – audit an existing sklearn Pipeline
# ===================================================================


class PipelineAuditor:
    """Audit an *existing* :class:`~sklearn.pipeline.Pipeline` object.

    Unlike :class:`AuditedPipeline`, which wraps steps at construction,
    ``PipelineAuditor`` takes an already-built (and possibly already-fitted)
    pipeline and creates an audit trail by replaying or intercepting
    its steps.

    Parameters
    ----------
    pipeline
        An sklearn ``Pipeline`` instance.
    audit_log
        Optional shared audit log.
    """

    def __init__(
        self,
        pipeline: Any,
        audit_log: Optional[_AuditLog] = None,
    ) -> None:
        self._pipeline = pipeline
        self._audit_log = audit_log or _default_audit_log
        self._step_records: List[_AuditRecord] = []

    @property
    def pipeline(self) -> Any:
        """The original pipeline being audited."""
        return self._pipeline

    @property
    def records(self) -> List[_AuditRecord]:
        """Return audit records collected during replay."""
        return list(self._step_records)

    def audit_fit(
        self,
        X: Any,
        y: Any = None,
        *,
        origin: Optional[Origin] = None,
        **fit_params: Any,
    ) -> Any:
        """Replay ``fit`` through the pipeline, recording each step.

        Parameters
        ----------
        X
            Training feature matrix.
        y
            Training target vector.
        origin
            Data partition label for provenance tracking.
        **fit_params
            Additional fit parameters forwarded to each step.

        Returns
        -------
        The fitted pipeline.
        """
        Xt = X
        steps = getattr(self._pipeline, "steps", [])

        for idx, (step_name, estimator) in enumerate(steps[:-1]):
            t0 = time.perf_counter()
            if hasattr(estimator, "fit_transform"):
                Xt_new = estimator.fit_transform(Xt, y, **fit_params)
            else:
                estimator.fit(Xt, y, **fit_params)
                Xt_new = estimator.transform(Xt)
            elapsed = (time.perf_counter() - t0) * 1000.0

            rec = self._make_record(
                step_name, estimator, "fit_transform", Xt, Xt_new, elapsed, origin,
            )
            self._step_records.append(rec)
            self._audit_log.record(rec)
            Xt = Xt_new

        if steps:
            final_name, final_estimator = steps[-1]
            t0 = time.perf_counter()
            final_estimator.fit(Xt, y, **fit_params)
            elapsed = (time.perf_counter() - t0) * 1000.0

            rec = self._make_record(
                final_name, final_estimator, "fit", Xt, Xt, elapsed, origin,
            )
            self._step_records.append(rec)
            self._audit_log.record(rec)

        return self._pipeline

    def audit_predict(
        self,
        X: Any,
        *,
        origin: Optional[Origin] = None,
        **params: Any,
    ) -> Any:
        """Replay ``predict`` through the pipeline, recording each step.

        Parameters
        ----------
        X
            Feature matrix for prediction.
        origin
            Data partition label for provenance tracking.

        Returns
        -------
        Predictions from the final estimator.
        """
        Xt = X
        steps = getattr(self._pipeline, "steps", [])

        for step_name, estimator in steps[:-1]:
            if not hasattr(estimator, "transform"):
                continue
            t0 = time.perf_counter()
            Xt_new = estimator.transform(Xt, **params)
            elapsed = (time.perf_counter() - t0) * 1000.0

            rec = self._make_record(
                step_name, estimator, "transform", Xt, Xt_new, elapsed, origin,
            )
            self._step_records.append(rec)
            self._audit_log.record(rec)
            Xt = Xt_new

        if steps:
            final_name, final_estimator = steps[-1]
            t0 = time.perf_counter()
            result = final_estimator.predict(Xt, **params)
            elapsed = (time.perf_counter() - t0) * 1000.0

            rec = self._make_record(
                final_name, final_estimator, "predict", Xt, result, elapsed, origin,
            )
            self._step_records.append(rec)
            self._audit_log.record(rec)
            return result

        return None

    def inspect_steps(self) -> List[Dict[str, Any]]:
        """Return a summary of each pipeline step without running it.

        Returns
        -------
        list of dict
            One dict per step with keys ``name``, ``class``, ``is_fitted``,
            ``profile``, and ``may_leak``.
        """
        result: List[Dict[str, Any]] = []
        steps = getattr(self._pipeline, "steps", [])
        for step_name, estimator in steps:
            profile = EstimatorClassifier.classify(estimator)
            is_fitted = hasattr(estimator, "n_features_in_") or hasattr(estimator, "classes_")
            result.append({
                "name": step_name,
                "class": type(estimator).__name__,
                "is_fitted": is_fitted,
                "profile": profile,
                "may_leak": profile.leaks_on_fit,
            })
        return result

    @staticmethod
    def _make_record(
        step_name: str,
        estimator: Any,
        method: str,
        X_in: Any,
        X_out: Any,
        wall_ms: float,
        origin: Optional[Origin],
    ) -> _AuditRecord:
        est_class = type(estimator).__name__
        op_map = {
            "fit": OpType.FIT,
            "transform": OpType.TRANSFORM_SK,
            "fit_transform": OpType.FIT_TRANSFORM,
            "predict": OpType.PREDICT,
            "predict_proba": OpType.PREDICT_PROBA,
            "score": OpType.SCORE,
        }
        op_type = op_map.get(method, OpType.UNKNOWN)

        in_shape = _extract_shape(X_in)
        out_shape = _extract_shape(X_out) if X_out is not None else in_shape

        provenance = None
        if origin is not None:
            test_frac = 1.0 if origin == Origin.TEST else 0.0
            provenance = ProvenanceInfo(
                test_fraction=test_frac,
                origin_set=frozenset({origin}),
            )

        return _AuditRecord(
            node_id=_generate_node_id(f"audit_{step_name}"),
            op_type=op_type,
            estimator_class=est_class,
            method=method,
            input_shape=in_shape,
            output_shape=out_shape,
            origin=origin,
            source_location=_caller_location(depth=4),
            wall_time_ms=wall_ms,
            provenance=provenance,
            metadata={"pipeline_step": step_name},
        )
