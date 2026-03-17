"""
taintflow.patterns.temporal – Temporal leakage pattern detectors.

Temporal leakage occurs when information from the future is used to
predict the past.  Common manifestations include:

* Time-series data sorted incorrectly (or shuffled) before splitting.
* Rolling-window / expanding-window features computed on the **full**
  dataset instead of only on the training partition.
* Lagged features that peek into future rows relative to the prediction
  timestamp.
* Seasonal patterns that repeat across train/test boundaries in a way
  that leaks holdout information.

The entry-point is :class:`TemporalLeakageDetector`, which orchestrates
the specialised sub-detectors and returns a list of
:class:`TemporalLeakagePattern` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from taintflow.core.types import OpType, Origin, Severity

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG
    from taintflow.dag.node import PipelineNode
    from taintflow.dag.edge import PipelineEdge


# ===================================================================
#  Supporting enumerations
# ===================================================================


class TemporalViolationType(Enum):
    """Specific kind of temporal ordering violation."""

    SORT_BEFORE_SPLIT = auto()
    SHUFFLE_BEFORE_SPLIT = auto()
    ROLLING_ON_FULL = auto()
    EXPANDING_ON_FULL = auto()
    EWM_ON_FULL = auto()
    RESAMPLE_ON_FULL = auto()
    LAG_PEEK_AHEAD = auto()
    DIFF_PEEK_AHEAD = auto()
    SHIFT_PEEK_AHEAD = auto()
    SEASONAL_CROSS_BOUNDARY = auto()
    NON_TEMPORAL_SPLIT = auto()


# ===================================================================
#  Pattern dataclass
# ===================================================================


@dataclass
class TemporalLeakagePattern:
    """A detected instance of temporal leakage in the pipeline DAG.

    Attributes
    ----------
    description : str
        Human-readable explanation of the detected pattern.
    severity : Severity
        Estimated severity of the leakage.
    source_nodes : list[str]
        Node IDs in the PI-DAG where the leakage originates.
    affected_features : list[str]
        Column names that are contaminated by the leakage.
    violation_type : TemporalViolationType
        Classification of the violation.
    temporal_gap : float
        Estimated number of time-steps of look-ahead (0 if not applicable).
    remediation : str
        Suggested fix for the detected leakage.
    """

    description: str
    severity: Severity
    source_nodes: List[str]
    affected_features: List[str]
    violation_type: TemporalViolationType = TemporalViolationType.SORT_BEFORE_SPLIT
    temporal_gap: float = 0.0
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "description": self.description,
            "severity": self.severity.value,
            "source_nodes": list(self.source_nodes),
            "affected_features": list(self.affected_features),
            "violation_type": self.violation_type.name,
            "temporal_gap": self.temporal_gap,
            "remediation": self.remediation,
        }


# ===================================================================
#  TimeSeriesValidator
# ===================================================================


class TimeSeriesValidator:
    """Verify that a train/test split respects temporal ordering.

    Walks backward from each partition (split) node and checks whether
    the data was sorted by a time column before the split occurred, and
    whether any shuffle operation intervened.
    """

    # Operations that indicate a temporal sort
    _SORT_OPS: frozenset[OpType] = frozenset({OpType.SORT_VALUES, OpType.SORT_INDEX})

    # Operations that destroy temporal ordering
    _SHUFFLE_OPS: frozenset[OpType] = frozenset({OpType.SAMPLE})

    # Splitting operations
    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    def validate(self, dag: PIDAG) -> List[TemporalLeakagePattern]:
        """Return patterns for every split node whose ancestors violate
        temporal ordering constraints."""
        patterns: List[TemporalLeakagePattern] = []
        for node in dag.nodes.values():
            if node.op_type not in self._SPLIT_OPS:
                continue
            patterns.extend(self._check_split_node(dag, node))
        return patterns

    def _check_split_node(
        self, dag: PIDAG, split_node: PipelineNode
    ) -> List[TemporalLeakagePattern]:
        """Inspect the ancestors of *split_node* for ordering violations."""
        results: List[TemporalLeakagePattern] = []
        ancestor_ids = dag.ancestors(split_node.node_id)
        has_sort = False
        has_shuffle_after_sort = False
        shuffle_node_id: Optional[str] = None

        topo = dag.topological_order()
        ancestor_ordered = [nid for nid in topo if nid in ancestor_ids]

        sort_index: int = -1
        for idx, nid in enumerate(ancestor_ordered):
            anc_node = dag.get_node(nid)
            if anc_node.op_type in self._SORT_OPS:
                has_sort = True
                sort_index = idx

        for idx, nid in enumerate(ancestor_ordered):
            anc_node = dag.get_node(nid)
            if anc_node.op_type in self._SHUFFLE_OPS and idx > sort_index and has_sort:
                has_shuffle_after_sort = True
                shuffle_node_id = nid
                break

        time_col = self._find_time_column(dag, split_node)
        affected = [time_col] if time_col else list(split_node.output_columns)

        if not has_sort:
            is_temporal = self._looks_temporal(dag, split_node)
            if is_temporal:
                results.append(TemporalLeakagePattern(
                    description=(
                        f"Split node '{split_node.node_id}' operates on data "
                        f"that appears temporal but was never sorted by time."
                    ),
                    severity=Severity.WARNING,
                    source_nodes=[split_node.node_id],
                    affected_features=affected,
                    violation_type=TemporalViolationType.NON_TEMPORAL_SPLIT,
                    remediation="Sort data by the time column before splitting.",
                ))

        if has_shuffle_after_sort and shuffle_node_id is not None:
            results.append(TemporalLeakagePattern(
                description=(
                    f"Data was sorted temporally but then shuffled at node "
                    f"'{shuffle_node_id}' before split '{split_node.node_id}'."
                ),
                severity=Severity.CRITICAL,
                source_nodes=[shuffle_node_id, split_node.node_id],
                affected_features=affected,
                violation_type=TemporalViolationType.SHUFFLE_BEFORE_SPLIT,
                remediation=(
                    "Remove the shuffle operation or move it after the split."
                ),
            ))

        return results

    @staticmethod
    def _find_time_column(dag: PIDAG, node: PipelineNode) -> Optional[str]:
        """Heuristically identify a datetime / timestamp column."""
        for col in node.input_schema:
            if col.dtype in ("datetime64[ns]", "datetime64", "timestamp"):
                return col.name
            lower_name = col.name.lower()
            if any(kw in lower_name for kw in ("date", "time", "timestamp", "dt")):
                return col.name
        return None

    @staticmethod
    def _looks_temporal(dag: PIDAG, split_node: PipelineNode) -> bool:
        """Return *True* if the data reaching *split_node* appears to contain
        a time column, suggesting temporal awareness is needed."""
        for col in split_node.input_schema:
            if col.dtype in ("datetime64[ns]", "datetime64", "timestamp"):
                return True
            lower = col.name.lower()
            if any(kw in lower for kw in ("date", "time", "timestamp", "year", "month")):
                return True
        return False


# ===================================================================
#  LookAheadDetector
# ===================================================================


class LookAheadDetector:
    """Detect features that use future values (look-ahead bias).

    Targets ``shift``, ``diff``, and ``pct_change`` operations where a
    negative period (or default that includes future rows) is applied
    **before** splitting.
    """

    _LOOKAHEAD_OPS: frozenset[OpType] = frozenset({
        OpType.DIFF, OpType.PCT_CHANGE,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
    })

    def detect(self, dag: PIDAG) -> List[TemporalLeakagePattern]:
        """Scan the DAG for look-ahead feature computations."""
        patterns: List[TemporalLeakagePattern] = []
        split_ids = {
            n.node_id for n in dag.nodes.values()
            if n.op_type in self._SPLIT_OPS
        }
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if node.op_type not in self._LOOKAHEAD_OPS:
                continue
            if self._is_before_any_split(dag, node.node_id, split_ids):
                period = node.metadata.get("periods", node.metadata.get("period", 1))
                if self._period_peeks_ahead(period):
                    affected = self._affected_columns(node)
                    patterns.append(TemporalLeakagePattern(
                        description=(
                            f"Node '{node.node_id}' ({node.op_type.name}) uses a "
                            f"negative period ({period}), peeking into future rows."
                        ),
                        severity=Severity.CRITICAL,
                        source_nodes=[node.node_id],
                        affected_features=affected,
                        violation_type=TemporalViolationType.LAG_PEEK_AHEAD
                        if node.op_type == OpType.DIFF
                        else TemporalViolationType.DIFF_PEEK_AHEAD,
                        temporal_gap=abs(float(period)),
                        remediation=(
                            "Use a positive period so that only past values are "
                            "referenced."
                        ),
                    ))
            if node.op_type in self._LOOKAHEAD_OPS:
                shift_dir = node.metadata.get("shift_direction", None)
                if shift_dir == "forward":
                    affected = self._affected_columns(node)
                    patterns.append(TemporalLeakagePattern(
                        description=(
                            f"Node '{node.node_id}' explicitly shifts data forward "
                            f"in time, using future values for current predictions."
                        ),
                        severity=Severity.CRITICAL,
                        source_nodes=[node.node_id],
                        affected_features=affected,
                        violation_type=TemporalViolationType.SHIFT_PEEK_AHEAD,
                        remediation="Use backward-looking shifts only.",
                    ))
        return patterns

    @staticmethod
    def _period_peeks_ahead(period: Any) -> bool:
        """Return *True* if *period* indicates a look-ahead."""
        try:
            return int(period) < 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_before_any_split(
        dag: PIDAG, node_id: str, split_ids: Set[str]
    ) -> bool:
        """Return *True* if *node_id* is an ancestor of any split node."""
        descendants = dag.descendants(node_id)
        return bool(descendants & split_ids)

    @staticmethod
    def _affected_columns(node: PipelineNode) -> List[str]:
        """Determine which columns a node affects."""
        target_cols = node.metadata.get("columns", None)
        if target_cols and isinstance(target_cols, (list, set, frozenset)):
            return sorted(target_cols)
        return sorted(node.output_columns)


# ===================================================================
#  RollingWindowAuditor
# ===================================================================


class RollingWindowAuditor:
    """Verify that rolling / expanding / EWM computations respect
    temporal boundaries (i.e. are applied *after* splitting)."""

    _WINDOW_OPS: frozenset[OpType] = frozenset({
        OpType.ROLLING, OpType.EXPANDING, OpType.EWM, OpType.RESAMPLE,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
    })

    def audit(self, dag: PIDAG) -> List[TemporalLeakagePattern]:
        """Return patterns for rolling-window operations on unsplit data."""
        patterns: List[TemporalLeakagePattern] = []
        split_ids = {
            n.node_id for n in dag.nodes.values()
            if n.op_type in self._SPLIT_OPS
        }
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if node.op_type not in self._WINDOW_OPS:
                continue
            if self._is_before_any_split(dag, node.node_id, split_ids):
                violation = self._classify_violation(node.op_type)
                window_size = node.metadata.get("window", "unknown")
                affected = sorted(node.output_columns)
                patterns.append(TemporalLeakagePattern(
                    description=(
                        f"Rolling/expanding operation '{node.op_type.name}' at "
                        f"node '{node.node_id}' (window={window_size}) is "
                        f"applied before the train/test split."
                    ),
                    severity=Severity.CRITICAL,
                    source_nodes=[node.node_id],
                    affected_features=affected,
                    violation_type=violation,
                    remediation=(
                        "Move the windowed aggregation after the split so that "
                        "the rolling window only sees training data."
                    ),
                ))
        return patterns

    @staticmethod
    def _is_before_any_split(
        dag: PIDAG, node_id: str, split_ids: Set[str]
    ) -> bool:
        """Return *True* if *node_id* is an ancestor of any split node."""
        descendants = dag.descendants(node_id)
        return bool(descendants & split_ids)

    @staticmethod
    def _classify_violation(op: OpType) -> TemporalViolationType:
        """Map an OpType to the matching violation enum."""
        mapping = {
            OpType.ROLLING: TemporalViolationType.ROLLING_ON_FULL,
            OpType.EXPANDING: TemporalViolationType.EXPANDING_ON_FULL,
            OpType.EWM: TemporalViolationType.EWM_ON_FULL,
            OpType.RESAMPLE: TemporalViolationType.RESAMPLE_ON_FULL,
        }
        return mapping.get(op, TemporalViolationType.ROLLING_ON_FULL)


# ===================================================================
#  SeasonalLeakageDetector
# ===================================================================


class SeasonalLeakageDetector:
    """Detect seasonal patterns that leak across train/test boundaries.

    If a pipeline extracts seasonal features (month-of-year, day-of-week,
    etc.) from the raw timestamp *before* splitting, and the split is not
    temporally contiguous, values from the test period may share identical
    seasonal buckets with training rows, creating a subtle information leak.
    """

    _DT_OPS: frozenset[OpType] = frozenset({OpType.DT_ACCESSOR})

    _SEASONAL_KEYWORDS: frozenset[str] = frozenset({
        "month", "dayofweek", "day_of_week", "weekday", "quarter",
        "week", "weekofyear", "day_of_year", "dayofyear", "hour",
        "season",
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
    })

    def detect(self, dag: PIDAG) -> List[TemporalLeakagePattern]:
        """Return patterns for seasonal feature extraction before splits."""
        patterns: List[TemporalLeakagePattern] = []
        split_ids = {
            n.node_id for n in dag.nodes.values()
            if n.op_type in self._SPLIT_OPS
        }
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if not self._is_seasonal_extraction(node):
                continue
            if self._is_ancestor_of_any(dag, node.node_id, split_ids):
                seasonal_cols = self._identify_seasonal_columns(node)
                patterns.append(TemporalLeakagePattern(
                    description=(
                        f"Seasonal feature extraction at node "
                        f"'{node.node_id}' occurs before the train/test split. "
                        f"Shared seasonal buckets may leak information."
                    ),
                    severity=Severity.WARNING,
                    source_nodes=[node.node_id],
                    affected_features=seasonal_cols,
                    violation_type=TemporalViolationType.SEASONAL_CROSS_BOUNDARY,
                    remediation=(
                        "Extract seasonal features after splitting, or use a "
                        "temporally contiguous split to reduce leakage."
                    ),
                ))
        return patterns

    def _is_seasonal_extraction(self, node: PipelineNode) -> bool:
        """Return *True* if the node extracts seasonal components."""
        if node.op_type in self._DT_OPS:
            accessor = node.metadata.get("accessor", "")
            if any(kw in accessor.lower() for kw in self._SEASONAL_KEYWORDS):
                return True

        for col in node.output_schema:
            if any(kw in col.name.lower() for kw in self._SEASONAL_KEYWORDS):
                if col.name not in {c.name for c in node.input_schema}:
                    return True
        return False

    def _identify_seasonal_columns(self, node: PipelineNode) -> List[str]:
        """Return output columns that look like seasonal features."""
        cols: List[str] = []
        for col in node.output_schema:
            if any(kw in col.name.lower() for kw in self._SEASONAL_KEYWORDS):
                cols.append(col.name)
        if not cols:
            cols = sorted(node.output_columns - node.input_columns)
        return cols if cols else sorted(node.output_columns)

    @staticmethod
    def _is_ancestor_of_any(
        dag: PIDAG, node_id: str, target_ids: Set[str]
    ) -> bool:
        """Return *True* if *node_id* is an ancestor of any node in
        *target_ids*."""
        descendants = dag.descendants(node_id)
        return bool(descendants & target_ids)


# ===================================================================
#  TemporalSplitAnalyzer
# ===================================================================


class TemporalSplitAnalyzer:
    """Analyze the temporal structure of a train/test split.

    Inspects partition nodes and their provenance information to determine
    whether the split is temporally contiguous (ideal for time-series) or
    interleaved (high leakage risk).
    """

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    def analyze(self, dag: PIDAG) -> List[Dict[str, Any]]:
        """Return a list of analysis records, one per split node.

        Each record contains:
          * ``split_node_id`` – the node ID of the split.
          * ``is_temporal`` – whether the split appears temporal.
          * ``has_time_column`` – whether a time column was found.
          * ``time_column`` – name of the detected time column or ``None``.
          * ``train_origins`` – origin labels observed in training data.
          * ``test_origins`` – origin labels observed in test data.
          * ``warnings`` – list of human-readable warnings.
        """
        results: List[Dict[str, Any]] = []
        for node in dag.nodes.values():
            if node.op_type not in self._SPLIT_OPS:
                continue
            results.append(self._analyze_split(dag, node))
        return results

    def _analyze_split(
        self, dag: PIDAG, split_node: PipelineNode
    ) -> Dict[str, Any]:
        """Build the analysis record for one split node."""
        time_col = self._find_time_column(split_node)
        is_temporal = time_col is not None
        train_origins: Set[str] = set()
        test_origins: Set[str] = set()
        for col_name, prov in split_node.provenance.items():
            for o in prov.origin_set:
                if o == Origin.TRAIN:
                    train_origins.add(col_name)
                elif o == Origin.TEST:
                    test_origins.add(col_name)

        warnings: List[str] = []
        if is_temporal and split_node.op_type == OpType.TRAIN_TEST_SPLIT:
            shuffle_flag = split_node.metadata.get("shuffle", None)
            if shuffle_flag is True:
                warnings.append(
                    "train_test_split is called with shuffle=True on temporal data."
                )
        if not is_temporal:
            has_any_time = self._has_temporal_ancestor(dag, split_node)
            if has_any_time:
                warnings.append(
                    "Data appears temporal upstream but the split node does "
                    "not reference a time column directly."
                )

        return {
            "split_node_id": split_node.node_id,
            "is_temporal": is_temporal,
            "has_time_column": time_col is not None,
            "time_column": time_col,
            "train_origins": sorted(train_origins),
            "test_origins": sorted(test_origins),
            "warnings": warnings,
        }

    @staticmethod
    def _find_time_column(node: PipelineNode) -> Optional[str]:
        """Heuristically identify a datetime column."""
        for col in node.input_schema:
            if col.dtype in ("datetime64[ns]", "datetime64", "timestamp"):
                return col.name
            lower = col.name.lower()
            if any(kw in lower for kw in ("date", "time", "timestamp", "dt")):
                return col.name
        return None

    @staticmethod
    def _has_temporal_ancestor(dag: PIDAG, node: PipelineNode) -> bool:
        """Return *True* if any ancestor has a time-like column."""
        for anc_id in dag.ancestors(node.node_id):
            anc = dag.get_node(anc_id)
            for col in anc.output_schema:
                if col.dtype in ("datetime64[ns]", "datetime64", "timestamp"):
                    return True
                if any(kw in col.name.lower() for kw in ("date", "time", "timestamp")):
                    return True
        return False


# ===================================================================
#  TemporalLeakageDetector  (top-level orchestrator)
# ===================================================================


class TemporalLeakageDetector:
    """Orchestrator that runs all temporal-leakage sub-detectors on a
    PI-DAG and returns the combined list of detected patterns.

    Usage::

        detector = TemporalLeakageDetector()
        patterns = detector.detect(dag)
        for p in patterns:
            print(p.severity, p.description)
    """

    def __init__(self) -> None:
        self._validator = TimeSeriesValidator()
        self._lookahead = LookAheadDetector()
        self._rolling = RollingWindowAuditor()
        self._seasonal = SeasonalLeakageDetector()
        self._split_analyzer = TemporalSplitAnalyzer()

    @property
    def validator(self) -> TimeSeriesValidator:
        """Access the underlying :class:`TimeSeriesValidator`."""
        return self._validator

    @property
    def split_analyzer(self) -> TemporalSplitAnalyzer:
        """Access the underlying :class:`TemporalSplitAnalyzer`."""
        return self._split_analyzer

    def detect(self, dag: PIDAG) -> List[TemporalLeakagePattern]:
        """Run all sub-detectors and return the combined patterns.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[TemporalLeakagePattern]
            Detected temporal leakage patterns, ordered by severity
            (critical first).
        """
        patterns: List[TemporalLeakagePattern] = []
        patterns.extend(self._validator.validate(dag))
        patterns.extend(self._lookahead.detect(dag))
        patterns.extend(self._rolling.audit(dag))
        patterns.extend(self._seasonal.detect(dag))

        # Enrich from split analysis – add warnings for temporal splits
        split_analyses = self._split_analyzer.analyze(dag)
        for analysis in split_analyses:
            for warning in analysis.get("warnings", []):
                patterns.append(TemporalLeakagePattern(
                    description=warning,
                    severity=Severity.WARNING,
                    source_nodes=[analysis["split_node_id"]],
                    affected_features=[],
                    violation_type=TemporalViolationType.NON_TEMPORAL_SPLIT,
                    remediation="Review the split strategy for temporal data.",
                ))

        patterns.sort(key=lambda p: p.severity, reverse=True)
        return patterns

    def detect_with_analysis(
        self, dag: PIDAG
    ) -> Tuple[List[TemporalLeakagePattern], List[Dict[str, Any]]]:
        """Like :meth:`detect` but also returns the split analysis records.

        Returns
        -------
        tuple[list[TemporalLeakagePattern], list[dict]]
            ``(patterns, split_analyses)``
        """
        patterns = self.detect(dag)
        analyses = self._split_analyzer.analyze(dag)
        return patterns, analyses
