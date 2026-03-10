"""
Data Drift Detection and Alerting
===================================

Continuous data drift detection for pipeline quality monitoring. Detects
concept drift, schema drift, volume drift, and freshness drift, and
produces alerts when drift exceeds configurable thresholds.

Drift types:
  - **Concept drift**: Statistical distribution changes between data batches.
  - **Schema drift**: Column additions, removals, type changes.
  - **Volume drift**: Significant changes in data volume.
  - **Freshness drift**: Data becoming stale beyond SLA.
  - **Feature drift**: Individual feature distribution shifts.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
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
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Drift Types
# =====================================================================


class DriftType(Enum):
    """Classification of drift types."""
    CONCEPT = "concept"
    SCHEMA = "schema"
    VOLUME = "volume"
    FRESHNESS = "freshness"
    FEATURE = "feature"
    DISTRIBUTION = "distribution"


class DriftSeverity(Enum):
    """Severity of detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertAction(Enum):
    """Recommended action for a drift alert."""
    NO_ACTION = "no_action"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    REPAIR_SCHEMA = "repair_schema"
    REPAIR_DATA = "repair_data"
    FULL_REFRESH = "full_refresh"
    PAUSE_PIPELINE = "pause_pipeline"


# =====================================================================
# Drift Result Types
# =====================================================================


@dataclass
class ColumnDrift:
    """Drift information for a single column.

    Attributes
    ----------
    column_name : str
        Name of the column.
    drift_score : float
        Drift score from 0.0 (no drift) to 1.0 (complete drift).
    drift_type : DriftType
        Type of drift detected.
    old_mean : float | None
        Previous mean (for numeric columns).
    new_mean : float | None
        Current mean (for numeric columns).
    old_null_rate : float
        Previous null rate.
    new_null_rate : float
        Current null rate.
    statistical_test : str
        Name of the statistical test used.
    p_value : float | None
        P-value from the statistical test.
    details : str
        Human-readable description of the drift.
    """
    column_name: str = ""
    drift_score: float = 0.0
    drift_type: DriftType = DriftType.FEATURE
    old_mean: Optional[float] = None
    new_mean: Optional[float] = None
    old_null_rate: float = 0.0
    new_null_rate: float = 0.0
    statistical_test: str = ""
    p_value: Optional[float] = None
    details: str = ""

    @property
    def is_significant(self) -> bool:
        if self.p_value is not None:
            return self.p_value < 0.05
        return self.drift_score > 0.3

    def __repr__(self) -> str:
        return (
            f"ColumnDrift({self.column_name}, score={self.drift_score:.3f}, "
            f"type={self.drift_type.value})"
        )


@dataclass
class DriftResult:
    """Complete drift detection result.

    Attributes
    ----------
    column_drifts : dict[str, ColumnDrift]
        Per-column drift information.
    overall_score : float
        Overall drift score (max of column scores).
    is_significant : bool
        Whether the drift is statistically significant.
    drift_type : DriftType
        Primary type of drift detected.
    severity : DriftSeverity
        Severity classification.
    recommended_action : str
        Recommended action based on drift analysis.
    detection_time_ms : float
        Time spent detecting drift.
    details : str
        Summary of drift findings.
    """
    column_drifts: Dict[str, ColumnDrift] = field(default_factory=dict)
    overall_score: float = 0.0
    is_significant: bool = False
    drift_type: DriftType = DriftType.CONCEPT
    severity: DriftSeverity = DriftSeverity.NONE
    recommended_action: str = ""
    detection_time_ms: float = 0.0
    details: str = ""

    @property
    def drifted_columns(self) -> List[str]:
        return [
            name for name, drift in self.column_drifts.items()
            if drift.is_significant
        ]

    @property
    def max_column_drift(self) -> Optional[ColumnDrift]:
        if not self.column_drifts:
            return None
        return max(self.column_drifts.values(), key=lambda d: d.drift_score)

    def summary(self) -> str:
        lines = [
            f"DriftResult(score={self.overall_score:.3f}, "
            f"severity={self.severity.value}):",
            f"  Type: {self.drift_type.value}",
            f"  Significant: {self.is_significant}",
            f"  Drifted columns: {len(self.drifted_columns)}/{len(self.column_drifts)}",
            f"  Action: {self.recommended_action}",
        ]
        for name, drift in sorted(
            self.column_drifts.items(),
            key=lambda x: x[1].drift_score,
            reverse=True,
        )[:5]:
            lines.append(f"  {name}: score={drift.drift_score:.3f}")
        return "\n".join(lines)


@dataclass
class SchemaDrift:
    """Schema drift between two data versions.

    Attributes
    ----------
    added_columns : list[str]
        New columns.
    dropped_columns : list[str]
        Removed columns.
    type_changed_columns : list[tuple[str, str, str]]
        Columns with changed types (name, old, new).
    nullable_changed : list[tuple[str, bool, bool]]
        Columns with changed nullability.
    is_breaking : bool
        Whether this is a breaking schema change.
    """
    added_columns: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    type_changed_columns: List[Tuple[str, str, str]] = field(default_factory=list)
    nullable_changed: List[Tuple[str, bool, bool]] = field(default_factory=list)
    is_breaking: bool = False

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_columns
            or self.dropped_columns
            or self.type_changed_columns
            or self.nullable_changed
        )

    @property
    def change_count(self) -> int:
        return (
            len(self.added_columns)
            + len(self.dropped_columns)
            + len(self.type_changed_columns)
            + len(self.nullable_changed)
        )


@dataclass
class DriftTimeSeries:
    """Drift scores over time for trend analysis.

    Attributes
    ----------
    timestamps : list[datetime]
        When each measurement was taken.
    scores : list[float]
        Drift scores at each timestamp.
    column_name : str
        Column tracked (or "overall" for aggregate).
    trend : str
        Detected trend (increasing, decreasing, stable, volatile).
    """
    timestamps: List[datetime] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    column_name: str = "overall"
    trend: str = "stable"

    def add_point(self, timestamp: datetime, score: float) -> None:
        self.timestamps.append(timestamp)
        self.scores.append(score)
        self._update_trend()

    def _update_trend(self) -> None:
        if len(self.scores) < 3:
            self.trend = "stable"
            return

        recent = self.scores[-5:]
        if len(recent) < 3:
            recent = self.scores

        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0

        if avg_diff > 0.05:
            self.trend = "increasing"
        elif avg_diff < -0.05:
            self.trend = "decreasing"
        elif max(recent) - min(recent) > 0.3:
            self.trend = "volatile"
        else:
            self.trend = "stable"


@dataclass
class Alert:
    """A drift alert.

    Attributes
    ----------
    alert_id : str
        Unique alert identifier.
    drift_type : DriftType
        Type of drift.
    severity : DriftSeverity
        Alert severity.
    action : AlertAction
        Recommended action.
    message : str
        Human-readable alert message.
    column : str | None
        Specific column (if applicable).
    score : float
        Drift score that triggered the alert.
    timestamp : datetime
        When the alert was generated.
    """
    alert_id: str = ""
    drift_type: DriftType = DriftType.CONCEPT
    severity: DriftSeverity = DriftSeverity.MEDIUM
    action: AlertAction = AlertAction.INVESTIGATE
    message: str = ""
    column: Optional[str] = None
    score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"Alert({self.severity.value}: {self.message[:60]})"
        )


@dataclass
class AlertConfig:
    """Configuration for drift alerting.

    Attributes
    ----------
    score_threshold : float
        Minimum drift score to trigger an alert.
    severity_thresholds : dict[str, float]
        Score thresholds for each severity level.
    enabled_drift_types : set[DriftType]
        Which drift types to alert on.
    cooldown_seconds : int
        Minimum seconds between alerts for the same source.
    max_alerts_per_hour : int
        Maximum alerts per hour.
    """
    score_threshold: float = 0.3
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.6,
        "critical": 0.8,
    })
    enabled_drift_types: Set[DriftType] = field(
        default_factory=lambda: set(DriftType)
    )
    cooldown_seconds: int = 300
    max_alerts_per_hour: int = 100


# =====================================================================
# Column Profile (lightweight, for drift detection)
# =====================================================================


@dataclass
class ColumnProfile:
    """Statistical profile of a column for drift comparison.

    Attributes
    ----------
    column_name : str
        Column name.
    count : int
        Number of non-null values.
    null_count : int
        Number of null values.
    null_rate : float
        Fraction of null values.
    unique_count : int
        Number of unique values.
    mean : float | None
        Mean (numeric columns).
    std : float | None
        Standard deviation (numeric columns).
    min_value : Any
        Minimum value.
    max_value : Any
        Maximum value.
    percentiles : dict[int, float]
        Percentile values.
    value_distribution : dict[str, int]
        Top value counts.
    dtype : str
        Data type.
    """
    column_name: str = ""
    count: int = 0
    null_count: int = 0
    null_rate: float = 0.0
    unique_count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Any = None
    max_value: Any = None
    percentiles: Dict[int, float] = field(default_factory=dict)
    value_distribution: Dict[str, int] = field(default_factory=dict)
    dtype: str = "unknown"


# =====================================================================
# Drift Detector
# =====================================================================


class DriftDetector:
    """Continuous data drift detection and alerting.

    Detects concept drift, schema drift, volume drift, and freshness
    drift between data batches or time windows.

    Parameters
    ----------
    significance_level : float
        Statistical significance level for tests (default: 0.05).
    default_window_size : int
        Default sliding window size for drift detection.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        default_window_size: int = 1000,
    ) -> None:
        self._alpha = significance_level
        self._window_size = default_window_size
        self._alert_history: List[Alert] = []
        self._time_series: Dict[str, DriftTimeSeries] = defaultdict(DriftTimeSeries)

    # ── Concept Drift ─────────────────────────────────────────────

    def detect_concept_drift(
        self,
        old_data: Any,
        new_data: Any,
        window_size: Optional[int] = None,
    ) -> DriftResult:
        """Detect concept drift between two data batches.

        Compares statistical distributions of all columns between
        old_data and new_data.

        Parameters
        ----------
        old_data : dict or DataFrame-like
            Reference data.
        new_data : dict or DataFrame-like
            Current data.
        window_size : int, optional
            Sample window size.

        Returns
        -------
        DriftResult
        """
        start = time.time()
        ws = window_size or self._window_size

        old_cols = self._get_columns(old_data)
        new_cols = self._get_columns(new_data)
        common_cols = set(old_cols) & set(new_cols)

        column_drifts: Dict[str, ColumnDrift] = {}

        for col in sorted(common_cols):
            old_values = self._get_column_values(old_data, col, ws)
            new_values = self._get_column_values(new_data, col, ws)

            drift = self._compute_column_drift(col, old_values, new_values)
            column_drifts[col] = drift

        overall = max(
            (d.drift_score for d in column_drifts.values()), default=0.0
        )
        significant = any(d.is_significant for d in column_drifts.values())
        severity = self._score_to_severity(overall)
        action = self._severity_to_action(severity)

        elapsed = (time.time() - start) * 1000

        result = DriftResult(
            column_drifts=column_drifts,
            overall_score=overall,
            is_significant=significant,
            drift_type=DriftType.CONCEPT,
            severity=severity,
            recommended_action=action,
            detection_time_ms=elapsed,
            details=f"Analyzed {len(common_cols)} columns, {sum(1 for d in column_drifts.values() if d.is_significant)} drifted",
        )

        self._time_series["overall"].add_point(datetime.utcnow(), overall)

        return result

    # ── Schema Drift ──────────────────────────────────────────────

    def detect_schema_drift(
        self,
        old_schema: Any,
        new_schema: Any,
    ) -> SchemaDrift:
        """Detect schema drift between two schemas.

        Parameters
        ----------
        old_schema : Schema or dict
            Previous schema.
        new_schema : Schema or dict
            Current schema.

        Returns
        -------
        SchemaDrift
        """
        old_cols = self._extract_schema_columns(old_schema)
        new_cols = self._extract_schema_columns(new_schema)

        old_names = set(old_cols.keys())
        new_names = set(new_cols.keys())

        added = sorted(new_names - old_names)
        dropped = sorted(old_names - new_names)

        type_changed: List[Tuple[str, str, str]] = []
        nullable_changed: List[Tuple[str, bool, bool]] = []

        for name in sorted(old_names & new_names):
            old_info = old_cols[name]
            new_info = new_cols[name]

            if old_info.get("type") != new_info.get("type"):
                type_changed.append((
                    name,
                    str(old_info.get("type", "")),
                    str(new_info.get("type", "")),
                ))

            if old_info.get("nullable") != new_info.get("nullable"):
                nullable_changed.append((
                    name,
                    old_info.get("nullable", True),
                    new_info.get("nullable", True),
                ))

        is_breaking = bool(dropped) or bool(type_changed)

        return SchemaDrift(
            added_columns=added,
            dropped_columns=dropped,
            type_changed_columns=type_changed,
            nullable_changed=nullable_changed,
            is_breaking=is_breaking,
        )

    # ── Volume Drift ──────────────────────────────────────────────

    def detect_volume_drift(
        self,
        old_count: int,
        new_count: int,
        threshold: float = 0.2,
    ) -> bool:
        """Detect significant changes in data volume.

        Parameters
        ----------
        old_count : int
            Previous row count.
        new_count : int
            Current row count.
        threshold : float
            Relative change threshold (0.2 = 20%).

        Returns
        -------
        bool
            True if volume change exceeds threshold.
        """
        if old_count == 0:
            return new_count > 0

        change_ratio = abs(new_count - old_count) / old_count
        return change_ratio > threshold

    # ── Freshness Drift ───────────────────────────────────────────

    def detect_freshness_drift(
        self,
        last_update: datetime,
        sla: timedelta,
    ) -> bool:
        """Detect if data is stale beyond SLA.

        Parameters
        ----------
        last_update : datetime
            When the data was last updated.
        sla : timedelta
            Maximum acceptable data age.

        Returns
        -------
        bool
            True if data is stale.
        """
        age = datetime.utcnow() - last_update
        return age > sla

    # ── Drift Scoring ─────────────────────────────────────────────

    def compute_drift_score(
        self,
        old_profile: ColumnProfile,
        new_profile: ColumnProfile,
    ) -> float:
        """Compute a drift score between two column profiles.

        Parameters
        ----------
        old_profile : ColumnProfile
            Previous column profile.
        new_profile : ColumnProfile
            Current column profile.

        Returns
        -------
        float
            Drift score from 0.0 to 1.0.
        """
        scores: List[float] = []

        null_drift = abs(old_profile.null_rate - new_profile.null_rate)
        scores.append(null_drift)

        if old_profile.mean is not None and new_profile.mean is not None:
            if old_profile.std and old_profile.std > 0:
                mean_shift = abs(new_profile.mean - old_profile.mean) / old_profile.std
                scores.append(min(1.0, mean_shift / 3.0))

        if old_profile.std is not None and new_profile.std is not None:
            if old_profile.std > 0:
                std_ratio = new_profile.std / old_profile.std
                std_drift = abs(1.0 - std_ratio)
                scores.append(min(1.0, std_drift))

        old_unique_rate = old_profile.unique_count / max(old_profile.count, 1)
        new_unique_rate = new_profile.unique_count / max(new_profile.count, 1)
        scores.append(abs(old_unique_rate - new_unique_rate))

        if old_profile.value_distribution and new_profile.value_distribution:
            dist_drift = self._distribution_distance(
                old_profile.value_distribution,
                new_profile.value_distribution,
            )
            scores.append(dist_drift)

        return max(scores) if scores else 0.0

    # ── Time Series Tracking ──────────────────────────────────────

    def track_drift_over_time(
        self,
        profiles: List[ColumnProfile],
    ) -> DriftTimeSeries:
        """Track drift evolution over a sequence of profiles.

        Parameters
        ----------
        profiles : list[ColumnProfile]
            Ordered list of profiles (oldest first).

        Returns
        -------
        DriftTimeSeries
        """
        if len(profiles) < 2:
            return DriftTimeSeries()

        ts = DriftTimeSeries(
            column_name=profiles[0].column_name if profiles else "unknown",
        )

        for i in range(1, len(profiles)):
            score = self.compute_drift_score(profiles[i-1], profiles[i])
            ts.add_point(datetime.utcnow(), score)

        return ts

    # ── Alerting ──────────────────────────────────────────────────

    def alert_on_drift(
        self,
        drift_result: DriftResult,
        alert_config: Optional[AlertConfig] = None,
    ) -> Optional[Alert]:
        """Generate an alert if drift exceeds configured thresholds.

        Parameters
        ----------
        drift_result : DriftResult
            The drift detection result.
        alert_config : AlertConfig, optional
            Alert configuration.

        Returns
        -------
        Alert | None
            An alert if triggered, None otherwise.
        """
        config = alert_config or AlertConfig()

        if drift_result.drift_type not in config.enabled_drift_types:
            return None

        if drift_result.overall_score < config.score_threshold:
            return None

        if self._in_cooldown(config):
            return None

        severity = self._compute_alert_severity(
            drift_result.overall_score, config
        )
        action = self._severity_to_alert_action(severity, drift_result.drift_type)

        max_col = drift_result.max_column_drift
        col_info = ""
        if max_col:
            col_info = f" (worst: {max_col.column_name}={max_col.drift_score:.3f})"

        alert = Alert(
            alert_id=hashlib.sha256(
                f"{datetime.utcnow().isoformat()}{drift_result.overall_score}".encode()
            ).hexdigest()[:12],
            drift_type=drift_result.drift_type,
            severity=severity,
            action=action,
            message=(
                f"Data drift detected: overall_score={drift_result.overall_score:.3f}, "
                f"{len(drift_result.drifted_columns)} drifted columns{col_info}"
            ),
            column=max_col.column_name if max_col else None,
            score=drift_result.overall_score,
        )

        self._alert_history.append(alert)
        return alert

    # ── Internal Helpers ──────────────────────────────────────────

    def _compute_column_drift(
        self,
        column_name: str,
        old_values: np.ndarray,
        new_values: np.ndarray,
    ) -> ColumnDrift:
        """Compute drift for a single column."""
        drift = ColumnDrift(column_name=column_name)

        old_nulls = np.isnan(old_values.astype(float)) if old_values.dtype.kind == 'f' else np.array([v is None for v in old_values])
        new_nulls = np.isnan(new_values.astype(float)) if new_values.dtype.kind == 'f' else np.array([v is None for v in new_values])

        old_null_rate = float(np.mean(old_nulls)) if len(old_nulls) > 0 else 0.0
        new_null_rate = float(np.mean(new_nulls)) if len(new_nulls) > 0 else 0.0
        drift.old_null_rate = old_null_rate
        drift.new_null_rate = new_null_rate

        try:
            old_numeric = old_values[~old_nulls].astype(float)
            new_numeric = new_values[~new_nulls].astype(float)

            if len(old_numeric) > 0 and len(new_numeric) > 0:
                drift.old_mean = float(np.mean(old_numeric))
                drift.new_mean = float(np.mean(new_numeric))

                old_std = float(np.std(old_numeric))
                new_std = float(np.std(new_numeric))

                drift.statistical_test = "ks_approximation"
                drift.drift_score = self._ks_approximation(
                    old_numeric, new_numeric
                )

                if old_std > 0:
                    mean_shift = abs(drift.new_mean - drift.old_mean) / old_std
                    drift.drift_score = max(
                        drift.drift_score,
                        min(1.0, mean_shift / 3.0),
                    )

                drift.p_value = max(0.0, 1.0 - drift.drift_score)

        except (ValueError, TypeError):
            drift.drift_score = abs(old_null_rate - new_null_rate)
            drift.statistical_test = "null_rate_comparison"

        drift.drift_score = max(
            drift.drift_score,
            abs(old_null_rate - new_null_rate),
        )

        return drift

    @staticmethod
    def _ks_approximation(old: np.ndarray, new: np.ndarray) -> float:
        """Approximate two-sample KS statistic."""
        if len(old) == 0 or len(new) == 0:
            return 0.0

        all_values = np.sort(np.concatenate([old, new]))
        n1 = len(old)
        n2 = len(new)

        old_sorted = np.sort(old)
        new_sorted = np.sort(new)

        old_cdf = np.searchsorted(old_sorted, all_values, side='right') / n1
        new_cdf = np.searchsorted(new_sorted, all_values, side='right') / n2

        ks_stat = float(np.max(np.abs(old_cdf - new_cdf)))
        return ks_stat

    @staticmethod
    def _distribution_distance(
        dist1: Dict[str, int],
        dist2: Dict[str, int],
    ) -> float:
        """Compute distance between two value distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 0.0

        total1 = sum(dist1.values()) or 1
        total2 = sum(dist2.values()) or 1

        distance = 0.0
        for key in all_keys:
            p1 = dist1.get(key, 0) / total1
            p2 = dist2.get(key, 0) / total2
            distance += abs(p1 - p2)

        return min(1.0, distance / 2.0)

    @staticmethod
    def _get_columns(data: Any) -> List[str]:
        """Get column names from data."""
        if isinstance(data, dict):
            return list(data.keys())
        if hasattr(data, "columns"):
            return list(data.columns)
        return []

    @staticmethod
    def _get_column_values(
        data: Any,
        column: str,
        max_rows: int,
    ) -> np.ndarray:
        """Extract column values as numpy array."""
        if isinstance(data, dict):
            values = data.get(column, [])
            arr = np.asarray(values[:max_rows])
            return arr
        if hasattr(data, "__getitem__"):
            try:
                col = data[column]
                if hasattr(col, "to_numpy"):
                    return col.to_numpy()[:max_rows]
                return np.asarray(col)[:max_rows]
            except (KeyError, TypeError):
                pass
        return np.array([])

    @staticmethod
    def _extract_schema_columns(schema: Any) -> Dict[str, Dict[str, Any]]:
        """Extract column info from a schema."""
        cols: Dict[str, Dict[str, Any]] = {}

        if isinstance(schema, dict):
            for name, info in schema.items():
                if isinstance(info, dict):
                    cols[name] = info
                else:
                    cols[name] = {"type": str(info)}
        elif hasattr(schema, "columns"):
            for col in schema.columns:
                name = getattr(col, "name", str(col))
                cols[name] = {
                    "type": str(getattr(col, "sql_type", "unknown")),
                    "nullable": getattr(col, "nullable", True),
                }

        return cols

    @staticmethod
    def _score_to_severity(score: float) -> DriftSeverity:
        """Convert drift score to severity."""
        if score >= 0.8:
            return DriftSeverity.CRITICAL
        if score >= 0.6:
            return DriftSeverity.HIGH
        if score >= 0.4:
            return DriftSeverity.MEDIUM
        if score >= 0.2:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    @staticmethod
    def _severity_to_action(severity: DriftSeverity) -> str:
        """Convert severity to recommended action string."""
        return {
            DriftSeverity.NONE: "no_action",
            DriftSeverity.LOW: "monitor",
            DriftSeverity.MEDIUM: "investigate",
            DriftSeverity.HIGH: "repair",
            DriftSeverity.CRITICAL: "pause_and_repair",
        }.get(severity, "investigate")

    def _compute_alert_severity(
        self,
        score: float,
        config: AlertConfig,
    ) -> DriftSeverity:
        """Compute alert severity based on config thresholds."""
        thresholds = config.severity_thresholds
        if score >= thresholds.get("critical", 0.8):
            return DriftSeverity.CRITICAL
        if score >= thresholds.get("high", 0.6):
            return DriftSeverity.HIGH
        if score >= thresholds.get("medium", 0.4):
            return DriftSeverity.MEDIUM
        if score >= thresholds.get("low", 0.2):
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    @staticmethod
    def _severity_to_alert_action(
        severity: DriftSeverity,
        drift_type: DriftType,
    ) -> AlertAction:
        """Convert severity and drift type to recommended action."""
        if severity == DriftSeverity.CRITICAL:
            if drift_type == DriftType.SCHEMA:
                return AlertAction.REPAIR_SCHEMA
            return AlertAction.PAUSE_PIPELINE

        if severity == DriftSeverity.HIGH:
            if drift_type == DriftType.SCHEMA:
                return AlertAction.REPAIR_SCHEMA
            if drift_type == DriftType.CONCEPT:
                return AlertAction.FULL_REFRESH
            return AlertAction.REPAIR_DATA

        if severity == DriftSeverity.MEDIUM:
            return AlertAction.INVESTIGATE

        return AlertAction.MONITOR

    def _in_cooldown(self, config: AlertConfig) -> bool:
        """Check if we're in alert cooldown period."""
        if not self._alert_history:
            return False

        last_alert = self._alert_history[-1]
        age = (datetime.utcnow() - last_alert.timestamp).total_seconds()
        return age < config.cooldown_seconds


# =====================================================================
# Convenience Functions
# =====================================================================


def detect_drift(
    old_data: Any,
    new_data: Any,
    window_size: int = 1000,
) -> DriftResult:
    """Convenience: detect concept drift between two datasets."""
    detector = DriftDetector(default_window_size=window_size)
    return detector.detect_concept_drift(old_data, new_data)


def detect_schema_drift(
    old_schema: Any,
    new_schema: Any,
) -> SchemaDrift:
    """Convenience: detect schema drift."""
    detector = DriftDetector()
    return detector.detect_schema_drift(old_schema, new_schema)


def is_data_stale(
    last_update: datetime,
    max_age_hours: float = 24.0,
) -> bool:
    """Convenience: check if data is stale."""
    detector = DriftDetector()
    sla = timedelta(hours=max_age_hours)
    return detector.detect_freshness_drift(last_update, sla)
