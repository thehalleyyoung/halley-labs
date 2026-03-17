"""
taintflow.core.types – Core type definitions for the TaintFlow system.

This module defines the enumerations, dataclasses, and type aliases that
form the vocabulary of every other TaintFlow component.  All types support
serialisation via ``to_dict`` / ``from_dict`` round-tripping, rich
comparison, and thorough validation.
"""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

# ===================================================================
#  Enumerations
# ===================================================================


class Origin(enum.Enum):
    """Partition label for a data row / column."""

    TRAIN = "train"
    TEST = "test"
    EXTERNAL = "external"

    def __repr__(self) -> str:
        return f"Origin.{self.name}"

    @classmethod
    def from_str(cls, s: str) -> "Origin":
        mapping = {v.value: v for v in cls}
        normed = s.strip().lower()
        if normed in mapping:
            return mapping[normed]
        raise ValueError(f"Unknown origin: {s!r}; expected one of {list(mapping)}")

    @classmethod
    def all(cls) -> frozenset["Origin"]:
        return frozenset(cls)


class Severity(enum.Enum):
    """Severity rating of detected leakage."""

    NEGLIGIBLE = "negligible"
    WARNING = "warning"
    CRITICAL = "critical"

    def __repr__(self) -> str:
        return f"Severity.{self.name}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = [Severity.NEGLIGIBLE, Severity.WARNING, Severity.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: object) -> bool:
        return self == other or self.__lt__(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return not self.__le__(other)

    def __ge__(self, other: object) -> bool:
        return self == other or self.__gt__(other)

    @classmethod
    def from_bits(cls, bits: float, *, warn: float = 1.0, crit: float = 8.0) -> "Severity":
        if bits >= crit:
            return cls.CRITICAL
        if bits >= warn:
            return cls.WARNING
        return cls.NEGLIGIBLE


class AnalysisPhase(enum.Enum):
    """Phases of the TaintFlow analysis pipeline."""

    INSTRUMENTATION = "instrumentation"
    TRACE_COLLECTION = "trace_collection"
    DAG_CONSTRUCTION = "dag_construction"
    CAPACITY_ESTIMATION = "capacity_estimation"
    TAINT_PROPAGATION = "taint_propagation"
    FIXPOINT_ITERATION = "fixpoint_iteration"
    ATTRIBUTION = "attribution"
    REPORTING = "reporting"

    def __repr__(self) -> str:
        return f"AnalysisPhase.{self.name}"


class NodeKind(enum.Enum):
    """Kind of node in the pipeline DAG."""

    DATA_SOURCE = "data_source"
    TRANSFORM = "transform"
    ESTIMATOR_FIT = "estimator_fit"
    ESTIMATOR_PREDICT = "estimator_predict"
    ESTIMATOR_TRANSFORM = "estimator_transform"
    SPLIT = "split"
    MERGE = "merge"
    FEATURE_ENGINEERING = "feature_engineering"
    EVALUATION = "evaluation"
    SINK = "sink"
    UNKNOWN = "unknown"

    def __repr__(self) -> str:
        return f"NodeKind.{self.name}"


class EdgeKind(enum.Enum):
    """Kind of edge in the pipeline DAG."""

    DATA_FLOW = "data_flow"
    FIT_DEPENDENCY = "fit_dependency"
    PARAMETER_FLOW = "parameter_flow"
    CONTROL_FLOW = "control_flow"
    LABEL_FLOW = "label_flow"
    INDEX_FLOW = "index_flow"
    AUXILIARY = "auxiliary"

    def __repr__(self) -> str:
        return f"EdgeKind.{self.name}"


class OpType(enum.Enum):
    """Operation types recognised by the TaintFlow transfer functions.

    Covers pandas DataFrame ops, sklearn estimator methods, numpy array
    ops, and common preprocessing patterns.  80+ members.
    """

    # -- pandas I/O ----------------------------------------------------------
    READ_CSV = "read_csv"
    READ_PARQUET = "read_parquet"
    READ_JSON = "read_json"
    READ_EXCEL = "read_excel"
    READ_SQL = "read_sql"
    READ_HDF = "read_hdf"
    READ_FEATHER = "read_feather"
    TO_CSV = "to_csv"
    TO_PARQUET = "to_parquet"

    # -- pandas selection / indexing -----------------------------------------
    GETITEM = "getitem"
    SETITEM = "setitem"
    LOC = "loc"
    ILOC = "iloc"
    AT = "at"
    IAT = "iat"
    HEAD = "head"
    TAIL = "tail"
    SAMPLE = "sample"
    NLARGEST = "nlargest"
    NSMALLEST = "nsmallest"
    QUERY = "query"
    FILTER = "filter"
    WHERE = "where"
    MASK = "mask"

    # -- pandas column manipulation ------------------------------------------
    DROP = "drop"
    DROP_DUPLICATES = "drop_duplicates"
    RENAME = "rename"
    ASSIGN = "assign"
    INSERT = "insert"
    POP = "pop"
    REINDEX = "reindex"
    SET_INDEX = "set_index"
    RESET_INDEX = "reset_index"
    SORT_VALUES = "sort_values"
    SORT_INDEX = "sort_index"

    # -- pandas reshaping / combining ----------------------------------------
    MERGE = "merge"
    JOIN = "join"
    CONCAT = "concat"
    APPEND = "append"
    PIVOT = "pivot"
    PIVOT_TABLE = "pivot_table"
    MELT = "melt"
    STACK = "stack"
    UNSTACK = "unstack"
    EXPLODE = "explode"
    CROSSTAB = "crosstab"

    # -- pandas aggregation / groupby ----------------------------------------
    GROUPBY = "groupby"
    AGG = "agg"
    AGGREGATE = "aggregate"
    TRANSFORM = "transform"
    APPLY = "apply"
    APPLYMAP = "applymap"
    MAP = "map"
    ROLLING = "rolling"
    EXPANDING = "expanding"
    EWM = "ewm"
    RESAMPLE = "resample"
    VALUE_COUNTS = "value_counts"
    DESCRIBE = "describe"
    CORR = "corr"
    COV = "cov"
    CUMSUM = "cumsum"
    CUMPROD = "cumprod"
    CUMMAX = "cummax"
    CUMMIN = "cummin"
    DIFF = "diff"
    PCT_CHANGE = "pct_change"
    RANK = "rank"

    # -- pandas missing data -------------------------------------------------
    FILLNA = "fillna"
    DROPNA = "dropna"
    INTERPOLATE = "interpolate"
    ISNA = "isna"
    NOTNA = "notna"
    REPLACE = "replace"
    CLIP = "clip"

    # -- pandas string / categorical -----------------------------------------
    STR_ACCESSOR = "str_accessor"
    CAT_ACCESSOR = "cat_accessor"
    DT_ACCESSOR = "dt_accessor"
    ASTYPE = "astype"
    GET_DUMMIES = "get_dummies"
    FACTORIZE = "factorize"
    CUT = "cut"
    QCUT = "qcut"

    # -- sklearn estimator methods -------------------------------------------
    FIT = "fit"
    PREDICT = "predict"
    FIT_TRANSFORM = "fit_transform"
    TRANSFORM_SK = "transform_sk"
    PREDICT_PROBA = "predict_proba"
    PREDICT_LOG_PROBA = "predict_log_proba"
    DECISION_FUNCTION = "decision_function"
    SCORE = "score"
    FIT_PREDICT = "fit_predict"
    INVERSE_TRANSFORM = "inverse_transform"

    # -- sklearn preprocessing -----------------------------------------------
    STANDARD_SCALER = "standard_scaler"
    MINMAX_SCALER = "minmax_scaler"
    ROBUST_SCALER = "robust_scaler"
    NORMALIZER = "normalizer"
    LABEL_ENCODER = "label_encoder"
    ORDINAL_ENCODER = "ordinal_encoder"
    ONEHOT_ENCODER = "onehot_encoder"
    TARGET_ENCODER = "target_encoder"
    POLYNOMIAL_FEATURES = "polynomial_features"
    KBINS_DISCRETIZER = "kbins_discretizer"
    BINARIZER = "binarizer"
    IMPUTER = "imputer"
    KNN_IMPUTER = "knn_imputer"

    # -- sklearn model selection / splitting ---------------------------------
    TRAIN_TEST_SPLIT = "train_test_split"
    CROSS_VAL_SCORE = "cross_val_score"
    CROSS_VALIDATE = "cross_validate"
    KFOLD_SPLIT = "kfold_split"
    STRATIFIED_KFOLD = "stratified_kfold"
    GROUP_KFOLD = "group_kfold"
    LEAVE_ONE_OUT = "leave_one_out"

    # -- numpy operations ----------------------------------------------------
    NP_ARRAY = "np_array"
    NP_CONCATENATE = "np_concatenate"
    NP_VSTACK = "np_vstack"
    NP_HSTACK = "np_hstack"
    NP_COLUMN_STACK = "np_column_stack"
    NP_SPLIT = "np_split"
    NP_WHERE = "np_where"
    NP_UNIQUE = "np_unique"
    NP_ARGSORT = "np_argsort"
    NP_LOG = "np_log"
    NP_EXP = "np_exp"
    NP_MATMUL = "np_matmul"
    NP_DOT = "np_dot"
    NP_MEAN = "np_mean"
    NP_STD = "np_std"
    NP_VAR = "np_var"
    NP_MEDIAN = "np_median"
    NP_SUM = "np_sum"

    # -- high-level pipeline categories --------------------------------------
    DATA_SOURCE = "data_source"
    CROSS_VAL_SPLIT = "cross_val_split"
    ESTIMATOR_FIT = "estimator_fit"
    AGGREGATION = "aggregation"
    PIPELINE_STAGE = "pipeline_stage"
    COLUMN_TRANSFORMER = "column_transformer"
    FEATURE_UNION = "feature_union"
    IMPUTATION = "imputation"
    ENCODING = "encoding"
    SCALING = "scaling"
    NORMALIZATION = "normalization"
    SELECTION = "selection"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"

    # -- misc / generic ------------------------------------------------------
    COPY = "copy"
    DEEPCOPY = "deepcopy"
    IDENTITY = "identity"
    CUSTOM = "custom"
    UNKNOWN = "unknown"

    def __repr__(self) -> str:
        return f"OpType.{self.name}"

    @property
    def is_sklearn(self) -> bool:
        _sk = {
            "fit", "predict", "fit_transform", "transform_sk",
            "predict_proba", "predict_log_proba", "decision_function",
            "score", "fit_predict", "inverse_transform",
            "standard_scaler", "minmax_scaler", "robust_scaler",
            "normalizer", "label_encoder", "ordinal_encoder",
            "onehot_encoder", "target_encoder", "polynomial_features",
            "kbins_discretizer", "binarizer", "imputer", "knn_imputer",
            "train_test_split", "cross_val_score", "cross_validate",
            "kfold_split", "stratified_kfold", "group_kfold", "leave_one_out",
        }
        return self.value in _sk

    @property
    def is_pandas(self) -> bool:
        return not self.is_sklearn and not self.is_numpy and self.value not in (
            "copy", "deepcopy", "identity", "custom", "unknown",
        )

    @property
    def is_numpy(self) -> bool:
        return self.value.startswith("np_")

    @property
    def is_aggregation(self) -> bool:
        _agg = {
            "agg", "aggregate", "groupby", "value_counts", "describe",
            "corr", "cov", "cumsum", "cumprod", "cummax", "cummin",
            "rolling", "expanding", "ewm", "resample",
            "np_mean", "np_std", "np_var", "np_median", "np_sum",
        }
        return self.value in _agg

    @property
    def may_leak(self) -> bool:
        """Heuristic: operations that can channel test info into train."""
        _leak_risk = {
            "merge", "join", "concat", "append", "fillna", "interpolate",
            "fit", "fit_transform", "fit_predict",
            "groupby", "agg", "aggregate", "transform",
            "corr", "cov", "describe",
            "standard_scaler", "minmax_scaler", "robust_scaler",
            "normalizer", "imputer", "knn_imputer", "target_encoder",
            "cross_val_score", "cross_validate",
        }
        return self.value in _leak_risk


# ===================================================================
#  Dataclasses
# ===================================================================


@dataclass(frozen=True)
class ColumnSchema:
    """Schema descriptor for a single DataFrame column."""

    name: str
    dtype: str = "object"
    nullable: bool = True
    is_target: bool = False
    is_index: bool = False
    cardinality: int | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("Column name must be non-empty.")
        if self.cardinality is not None and self.cardinality < 0:
            errors.append(f"Cardinality must be >= 0, got {self.cardinality}.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "dtype": self.dtype, "nullable": self.nullable}
        if self.is_target:
            d["is_target"] = True
        if self.is_index:
            d["is_index"] = True
        if self.cardinality is not None:
            d["cardinality"] = self.cardinality
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnSchema":
        return cls(
            name=str(data["name"]),
            dtype=str(data.get("dtype", "object")),
            nullable=bool(data.get("nullable", True)),
            is_target=bool(data.get("is_target", False)),
            is_index=bool(data.get("is_index", False)),
            cardinality=data.get("cardinality"),
        )

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"dtype={self.dtype!r}"]
        if self.is_target:
            parts.append("TARGET")
        if self.is_index:
            parts.append("INDEX")
        return f"ColumnSchema({', '.join(parts)})"

    def entropy_bound(self) -> float:
        if self.cardinality is not None and self.cardinality > 0:
            return math.log2(self.cardinality)
        if self.dtype in ("bool", "boolean"):
            return 1.0
        if self.dtype.startswith("int"):
            try:
                bits = int(self.dtype.replace("int", ""))
                return float(bits)
            except ValueError:
                return 64.0
        if self.dtype.startswith("float"):
            try:
                bits = int(self.dtype.replace("float", ""))
                return float(bits)
            except ValueError:
                return 64.0
        return 64.0


@dataclass(frozen=True)
class ShapeMetadata:
    """Row / column counts, optionally split by partition."""

    n_rows: int
    n_cols: int
    n_test_rows: int = 0
    n_train_rows: int = 0
    n_external_rows: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_train_rows", self.n_train_rows or (self.n_rows - self.n_test_rows - self.n_external_rows))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.n_rows < 0:
            errors.append(f"n_rows must be >= 0, got {self.n_rows}")
        if self.n_cols < 0:
            errors.append(f"n_cols must be >= 0, got {self.n_cols}")
        if self.n_test_rows < 0:
            errors.append(f"n_test_rows must be >= 0, got {self.n_test_rows}")
        if self.n_test_rows + self.n_external_rows > self.n_rows:
            errors.append("n_test_rows + n_external_rows > n_rows")
        return errors

    @property
    def test_fraction(self) -> float:
        if self.n_rows == 0:
            return 0.0
        return self.n_test_rows / self.n_rows

    @property
    def train_fraction(self) -> float:
        if self.n_rows == 0:
            return 0.0
        return self.n_train_rows / self.n_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_test_rows": self.n_test_rows,
            "n_train_rows": self.n_train_rows,
            "n_external_rows": self.n_external_rows,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShapeMetadata":
        return cls(
            n_rows=int(data["n_rows"]),
            n_cols=int(data["n_cols"]),
            n_test_rows=int(data.get("n_test_rows", 0)),
            n_train_rows=int(data.get("n_train_rows", 0)),
            n_external_rows=int(data.get("n_external_rows", 0)),
        )

    def __repr__(self) -> str:
        return (
            f"Shape({self.n_rows}×{self.n_cols}, "
            f"test={self.n_test_rows}, train={self.n_train_rows})"
        )


@dataclass(frozen=True)
class ProvenanceInfo:
    """Provenance summary: which partitions contribute, and how much."""

    test_fraction: float  # ρ ∈ [0, 1]
    origin_set: FrozenSet[Origin] = field(default_factory=lambda: frozenset({Origin.TRAIN}))
    source_id: str = ""
    description: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not (0.0 <= self.test_fraction <= 1.0):
            errors.append(f"test_fraction (ρ) must be in [0,1], got {self.test_fraction}")
        if not self.origin_set:
            errors.append("origin_set must be non-empty")
        return errors

    @property
    def is_pure_train(self) -> bool:
        return self.origin_set == frozenset({Origin.TRAIN})

    @property
    def is_pure_test(self) -> bool:
        return self.origin_set == frozenset({Origin.TEST})

    @property
    def is_mixed(self) -> bool:
        return Origin.TRAIN in self.origin_set and Origin.TEST in self.origin_set

    @property
    def rho(self) -> float:
        return self.test_fraction

    def merge(self, other: "ProvenanceInfo") -> "ProvenanceInfo":
        merged_origins = self.origin_set | other.origin_set
        total = self.test_fraction + other.test_fraction
        avg_rho = total / 2.0
        return ProvenanceInfo(
            test_fraction=avg_rho,
            origin_set=merged_origins,
            source_id=f"{self.source_id}+{other.source_id}" if self.source_id and other.source_id else self.source_id or other.source_id,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "test_fraction": self.test_fraction,
            "origin_set": sorted(o.value for o in self.origin_set),
        }
        if self.source_id:
            d["source_id"] = self.source_id
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceInfo":
        origins = frozenset(Origin.from_str(s) for s in data.get("origin_set", ["train"]))
        return cls(
            test_fraction=float(data["test_fraction"]),
            origin_set=origins,
            source_id=str(data.get("source_id", "")),
            description=str(data.get("description", "")),
        )

    def __repr__(self) -> str:
        origins = ",".join(o.name for o in sorted(self.origin_set, key=lambda o: o.value))
        return f"Prov(ρ={self.test_fraction:.3f}, origins={{{origins}}})"


@dataclass(frozen=True)
class TaintLabel:
    """A taint label ℓ = (S, b) ∈ P({tr,te,ext}) × [0, B_max]."""

    origins: FrozenSet[Origin] = field(default_factory=frozenset)
    bit_bound: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.bit_bound < 0.0:
            errors.append(f"bit_bound must be >= 0, got {self.bit_bound}")
        if math.isnan(self.bit_bound):
            errors.append("bit_bound must not be NaN")
        return errors

    @property
    def is_clean(self) -> bool:
        return not self.origins or self.bit_bound == 0.0

    @property
    def is_test_tainted(self) -> bool:
        return Origin.TEST in self.origins and self.bit_bound > 0.0

    @property
    def severity(self) -> Severity:
        return Severity.from_bits(self.bit_bound)

    def join(self, other: "TaintLabel") -> "TaintLabel":
        return TaintLabel(
            origins=self.origins | other.origins,
            bit_bound=max(self.bit_bound, other.bit_bound),
        )

    def meet(self, other: "TaintLabel") -> "TaintLabel":
        return TaintLabel(
            origins=self.origins & other.origins,
            bit_bound=min(self.bit_bound, other.bit_bound),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origins": sorted(o.value for o in self.origins),
            "bit_bound": self.bit_bound,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaintLabel":
        origins = frozenset(Origin.from_str(s) for s in data.get("origins", []))
        return cls(origins=origins, bit_bound=float(data.get("bit_bound", 0.0)))

    def __repr__(self) -> str:
        origins = ",".join(o.name for o in sorted(self.origins, key=lambda o: o.value))
        return f"Taint({{{origins}}}, {self.bit_bound:.2f} bits)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaintLabel):
            return NotImplemented
        return self.origins == other.origins and math.isclose(self.bit_bound, other.bit_bound, abs_tol=1e-12)

    def __hash__(self) -> int:
        return hash((self.origins, round(self.bit_bound, 10)))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TaintLabel):
            return NotImplemented
        if self.origins < other.origins:
            return True
        if self.origins == other.origins:
            return self.bit_bound < other.bit_bound
        return False

    def __le__(self, other: object) -> bool:
        if not isinstance(other, TaintLabel):
            return NotImplemented
        return self.origins <= other.origins and self.bit_bound <= other.bit_bound + 1e-12


@dataclass
class FeatureLeakage:
    """Leakage report for a single feature (column)."""

    column_name: str
    bit_bound: float
    severity: Severity
    origins: FrozenSet[Origin]
    contributing_stages: List[str] = field(default_factory=list)
    remediation: str = ""
    explanation: str = ""
    confidence: float = 1.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.column_name:
            errors.append("column_name must be non-empty")
        if self.bit_bound < 0.0:
            errors.append(f"bit_bound must be >= 0, got {self.bit_bound}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence must be in [0,1], got {self.confidence}")
        return errors

    @property
    def is_critical(self) -> bool:
        return self.severity == Severity.CRITICAL

    @property
    def is_clean(self) -> bool:
        return self.severity == Severity.NEGLIGIBLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column_name,
            "bit_bound": self.bit_bound,
            "severity": self.severity.value,
            "origins": sorted(o.value for o in self.origins),
            "stages": self.contributing_stages,
            "remediation": self.remediation,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureLeakage":
        return cls(
            column_name=str(data["column"]),
            bit_bound=float(data["bit_bound"]),
            severity=Severity(data["severity"]),
            origins=frozenset(Origin.from_str(s) for s in data.get("origins", [])),
            contributing_stages=list(data.get("stages", [])),
            remediation=str(data.get("remediation", "")),
            explanation=str(data.get("explanation", "")),
            confidence=float(data.get("confidence", 1.0)),
        )

    def __repr__(self) -> str:
        return (
            f"FeatureLeakage({self.column_name!r}, "
            f"{self.bit_bound:.2f} bits, {self.severity.name})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureLeakage):
            return NotImplemented
        return (
            self.column_name == other.column_name
            and math.isclose(self.bit_bound, other.bit_bound, abs_tol=1e-12)
            and self.severity == other.severity
            and self.origins == other.origins
        )

    def __hash__(self) -> int:
        return hash((self.column_name, round(self.bit_bound, 10), self.severity))


@dataclass
class StageLeakage:
    """Leakage summary for a single pipeline stage (DAG node)."""

    stage_id: str
    stage_name: str
    op_type: OpType
    node_kind: NodeKind
    max_bit_bound: float
    mean_bit_bound: float
    feature_leakages: List[FeatureLeakage] = field(default_factory=list)
    severity: Severity = Severity.NEGLIGIBLE
    description: str = ""

    def __post_init__(self) -> None:
        if self.feature_leakages and self.severity == Severity.NEGLIGIBLE:
            max_sev = max((fl.severity for fl in self.feature_leakages), default=Severity.NEGLIGIBLE)
            object.__setattr__(self, "severity", max_sev) if hasattr(self, "__dataclass_fields__") else None
            self.severity = max_sev

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.stage_id:
            errors.append("stage_id must be non-empty")
        if self.max_bit_bound < 0:
            errors.append(f"max_bit_bound must be >= 0, got {self.max_bit_bound}")
        for fl in self.feature_leakages:
            errors.extend(fl.validate())
        return errors

    @property
    def n_leaking_features(self) -> int:
        return sum(1 for fl in self.feature_leakages if not fl.is_clean)

    @property
    def total_bit_bound(self) -> float:
        return sum(fl.bit_bound for fl in self.feature_leakages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "op_type": self.op_type.value,
            "node_kind": self.node_kind.value,
            "max_bit_bound": self.max_bit_bound,
            "mean_bit_bound": self.mean_bit_bound,
            "severity": self.severity.value,
            "features": [fl.to_dict() for fl in self.feature_leakages],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageLeakage":
        features = [FeatureLeakage.from_dict(f) for f in data.get("features", [])]
        return cls(
            stage_id=str(data["stage_id"]),
            stage_name=str(data["stage_name"]),
            op_type=OpType(data["op_type"]),
            node_kind=NodeKind(data["node_kind"]),
            max_bit_bound=float(data["max_bit_bound"]),
            mean_bit_bound=float(data["mean_bit_bound"]),
            feature_leakages=features,
            severity=Severity(data.get("severity", "negligible")),
            description=str(data.get("description", "")),
        )

    def __repr__(self) -> str:
        return (
            f"StageLeakage({self.stage_id!r}, {self.op_type.name}, "
            f"max={self.max_bit_bound:.2f} bits, {self.severity.name})"
        )


@dataclass
class LeakageReport:
    """Top-level report produced by a TaintFlow audit."""

    pipeline_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    overall_severity: Severity = Severity.NEGLIGIBLE
    total_bit_bound: float = 0.0
    n_stages: int = 0
    n_features: int = 0
    n_leaking_features: int = 0
    stage_leakages: List[StageLeakage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    analysis_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.stage_leakages:
            self.n_stages = len(self.stage_leakages)
            self.n_features = sum(len(sl.feature_leakages) for sl in self.stage_leakages)
            self.n_leaking_features = sum(sl.n_leaking_features for sl in self.stage_leakages)
            self.total_bit_bound = max((sl.max_bit_bound for sl in self.stage_leakages), default=0.0)
            self.overall_severity = max(
                (sl.severity for sl in self.stage_leakages),
                default=Severity.NEGLIGIBLE,
            )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.pipeline_name:
            errors.append("pipeline_name must be non-empty")
        for sl in self.stage_leakages:
            errors.extend(sl.validate())
        return errors

    @property
    def is_clean(self) -> bool:
        return self.overall_severity == Severity.NEGLIGIBLE

    @property
    def summary_line(self) -> str:
        return (
            f"{self.pipeline_name}: {self.overall_severity.name} "
            f"({self.total_bit_bound:.2f} bits, "
            f"{self.n_leaking_features}/{self.n_features} features)"
        )

    def stages_by_severity(self) -> list[StageLeakage]:
        order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.NEGLIGIBLE: 2}
        return sorted(self.stage_leakages, key=lambda s: (order.get(s.severity, 3), -s.max_bit_bound))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "timestamp": self.timestamp,
            "overall_severity": self.overall_severity.value,
            "total_bit_bound": self.total_bit_bound,
            "n_stages": self.n_stages,
            "n_features": self.n_features,
            "n_leaking_features": self.n_leaking_features,
            "stages": [sl.to_dict() for sl in self.stage_leakages],
            "metadata": self.metadata,
            "config_snapshot": self.config_snapshot,
            "analysis_duration_ms": self.analysis_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LeakageReport":
        stages = [StageLeakage.from_dict(s) for s in data.get("stages", [])]
        return cls(
            pipeline_name=str(data["pipeline_name"]),
            timestamp=str(data.get("timestamp", "")),
            stage_leakages=stages,
            metadata=dict(data.get("metadata", {})),
            config_snapshot=dict(data.get("config_snapshot", {})),
            analysis_duration_ms=float(data.get("analysis_duration_ms", 0.0)),
        )

    def fingerprint(self) -> str:
        import json
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"LeakageReport({self.summary_line})"


@dataclass(frozen=True)
class PipelineMetadata:
    """Metadata about the pipeline under analysis."""

    name: str
    source_file: str = ""
    n_stages: int = 0
    n_edges: int = 0
    libraries: Tuple[str, ...] = ()
    python_version: str = ""
    framework: str = ""
    description: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("Pipeline name must be non-empty")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "n_stages": self.n_stages,
            "n_edges": self.n_edges,
        }
        if self.source_file:
            d["source_file"] = self.source_file
        if self.libraries:
            d["libraries"] = list(self.libraries)
        if self.python_version:
            d["python_version"] = self.python_version
        if self.framework:
            d["framework"] = self.framework
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineMetadata":
        return cls(
            name=str(data["name"]),
            source_file=str(data.get("source_file", "")),
            n_stages=int(data.get("n_stages", 0)),
            n_edges=int(data.get("n_edges", 0)),
            libraries=tuple(data.get("libraries", ())),
            python_version=str(data.get("python_version", "")),
            framework=str(data.get("framework", "")),
            description=str(data.get("description", "")),
        )

    def __repr__(self) -> str:
        return f"PipelineMeta({self.name!r}, stages={self.n_stages}, edges={self.n_edges})"


@dataclass
class AnalysisConfig:
    """Per-analysis configuration extracted from global config."""

    b_max: float = 64.0
    alpha: float = 0.05
    max_iterations: int = 1000
    use_widening: bool = True
    widening_delay: int = 3
    use_narrowing: bool = True
    narrowing_iterations: int = 5
    epsilon: float = 1e-10
    track_provenance: bool = True
    parallel: bool = False
    n_workers: int = 1

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.b_max <= 0:
            errors.append(f"b_max must be > 0, got {self.b_max}")
        if not (0.0 < self.alpha < 1.0):
            errors.append(f"alpha must be in (0,1), got {self.alpha}")
        if self.max_iterations < 1:
            errors.append(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.widening_delay < 0:
            errors.append(f"widening_delay must be >= 0, got {self.widening_delay}")
        if self.epsilon <= 0:
            errors.append(f"epsilon must be > 0, got {self.epsilon}")
        if self.n_workers < 1:
            errors.append(f"n_workers must be >= 1, got {self.n_workers}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_max": self.b_max,
            "alpha": self.alpha,
            "max_iterations": self.max_iterations,
            "use_widening": self.use_widening,
            "widening_delay": self.widening_delay,
            "use_narrowing": self.use_narrowing,
            "narrowing_iterations": self.narrowing_iterations,
            "epsilon": self.epsilon,
            "track_provenance": self.track_provenance,
            "parallel": self.parallel,
            "n_workers": self.n_workers,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisConfig":
        return cls(
            b_max=float(data.get("b_max", 64.0)),
            alpha=float(data.get("alpha", 0.05)),
            max_iterations=int(data.get("max_iterations", 1000)),
            use_widening=bool(data.get("use_widening", True)),
            widening_delay=int(data.get("widening_delay", 3)),
            use_narrowing=bool(data.get("use_narrowing", True)),
            narrowing_iterations=int(data.get("narrowing_iterations", 5)),
            epsilon=float(data.get("epsilon", 1e-10)),
            track_provenance=bool(data.get("track_provenance", True)),
            parallel=bool(data.get("parallel", False)),
            n_workers=int(data.get("n_workers", 1)),
        )

    def __repr__(self) -> str:
        return (
            f"AnalysisConfig(B_max={self.b_max}, α={self.alpha}, "
            f"max_iter={self.max_iterations}, widen={self.use_widening})"
        )


@dataclass(frozen=True)
class ChannelParams:
    """Parameters describing an information-theoretic channel between stages."""

    capacity_bits: float = 0.0
    noise_variance: float = 0.0
    n_uses: int = 1
    channel_type: str = "deterministic"
    input_alphabet_size: int = 0
    output_alphabet_size: int = 0
    description: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.capacity_bits < 0:
            errors.append(f"capacity_bits must be >= 0, got {self.capacity_bits}")
        if self.noise_variance < 0:
            errors.append(f"noise_variance must be >= 0, got {self.noise_variance}")
        if self.n_uses < 1:
            errors.append(f"n_uses must be >= 1, got {self.n_uses}")
        return errors

    @property
    def total_capacity(self) -> float:
        return self.capacity_bits * self.n_uses

    @property
    def is_noiseless(self) -> bool:
        return self.noise_variance == 0.0

    @property
    def is_deterministic(self) -> bool:
        return self.channel_type == "deterministic"

    def attenuated_capacity(self, attenuation: float = 1.0) -> float:
        if attenuation <= 0.0:
            return 0.0
        return min(self.total_capacity * attenuation, self.total_capacity)

    def gaussian_capacity(self, signal_power: float) -> float:
        if self.noise_variance <= 0.0:
            return self.capacity_bits
        snr = signal_power / self.noise_variance
        return 0.5 * math.log2(1.0 + snr)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "capacity_bits": self.capacity_bits,
            "channel_type": self.channel_type,
        }
        if self.noise_variance > 0:
            d["noise_variance"] = self.noise_variance
        if self.n_uses > 1:
            d["n_uses"] = self.n_uses
        if self.input_alphabet_size > 0:
            d["input_alphabet_size"] = self.input_alphabet_size
        if self.output_alphabet_size > 0:
            d["output_alphabet_size"] = self.output_alphabet_size
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelParams":
        return cls(
            capacity_bits=float(data.get("capacity_bits", 0.0)),
            noise_variance=float(data.get("noise_variance", 0.0)),
            n_uses=int(data.get("n_uses", 1)),
            channel_type=str(data.get("channel_type", "deterministic")),
            input_alphabet_size=int(data.get("input_alphabet_size", 0)),
            output_alphabet_size=int(data.get("output_alphabet_size", 0)),
            description=str(data.get("description", "")),
        )

    def __repr__(self) -> str:
        return (
            f"Channel({self.channel_type}, C={self.capacity_bits:.2f} bits, "
            f"n={self.n_uses})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelParams):
            return NotImplemented
        return (
            math.isclose(self.capacity_bits, other.capacity_bits, abs_tol=1e-12)
            and math.isclose(self.noise_variance, other.noise_variance, abs_tol=1e-12)
            and self.n_uses == other.n_uses
            and self.channel_type == other.channel_type
        )

    def __hash__(self) -> int:
        return hash((
            round(self.capacity_bits, 10),
            round(self.noise_variance, 10),
            self.n_uses,
            self.channel_type,
        ))


# ===================================================================
#  Type aliases
# ===================================================================

ColumnName = str
NodeId = str
EdgeId = Tuple[NodeId, NodeId]
TaintMap = Dict[ColumnName, TaintLabel]
OriginSet = FrozenSet[Origin]
BitBound = float
StageId = str
FeatureName = str
SeverityLevel = Severity
ColumnSchemaList = List[ColumnSchema]
FeatureLeakageList = List[FeatureLeakage]
StageLeakageList = List[StageLeakage]
