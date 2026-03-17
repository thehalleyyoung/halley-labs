"""
taintflow.patterns.feature_leakage – Feature leakage pattern detectors.

Feature leakage occurs when feature selection, feature importance
calculation, or dimensionality reduction is performed on the **full**
(unsplit) dataset rather than on the training partition alone.  This
causes the selector to observe test-set statistics, inflating apparent
model performance and producing unreliable feature rankings.

Common manifestations include:

* ``SelectKBest``, ``VarianceThreshold``, or univariate filters fitted
  on the combined train+test data before splitting.
* Correlation-based filtering (Pearson, Spearman, mutual information)
  that sees the full joint distribution.
* Tree-based or model-based feature importance (random-forest, LASSO)
  estimated on the complete dataset.
* Dimensionality reduction (PCA, truncated SVD, UMAP) fitted on
  unsplit data, embedding test information into the reduced space.

The entry-point is :class:`FeatureLeakageDetector`, which orchestrates
the specialised sub-detectors and returns a list of
:class:`FeatureLeakagePattern` instances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

from taintflow.core.types import OpType, Origin, Severity

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG
    from taintflow.dag.node import PipelineNode
    from taintflow.dag.edge import PipelineEdge


# ===================================================================
#  Supporting enumerations
# ===================================================================


class FeatureViolationType(Enum):
    """Specific kind of feature-selection leakage violation."""

    SELECTION_BEFORE_SPLIT = auto()
    CORRELATION_FILTER_ON_FULL = auto()
    VARIANCE_THRESHOLD_ON_FULL = auto()
    MUTUAL_INFO_ON_FULL = auto()
    CHI2_ON_FULL = auto()
    PCA_ON_FULL = auto()
    FEATURE_IMPORTANCE_ON_FULL = auto()
    RECURSIVE_ELIMINATION_ON_FULL = auto()
    LASSO_SELECTION_ON_FULL = auto()
    BORUTA_ON_FULL = auto()
    STEPWISE_ON_FULL = auto()


# ===================================================================
#  Pattern dataclass
# ===================================================================


@dataclass
class FeatureLeakagePattern:
    """A detected instance of feature leakage in the pipeline DAG.

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
    violation_type : FeatureViolationType
        Classification of the violation.
    selected_features : list[str]
        Features that were selected (kept) by the leaking operation.
    eliminated_features : list[str]
        Features that were eliminated (dropped) by the leaking operation.
    selection_method : str
        Name of the selection / importance method that leaked.
    information_leaked_bits : float
        Estimated upper bound on the number of bits of test information
        that leaked into the selection decision.
    remediation : str
        Suggested fix for the detected leakage.
    """

    description: str
    severity: Severity
    source_nodes: List[str]
    affected_features: List[str]
    violation_type: FeatureViolationType = FeatureViolationType.SELECTION_BEFORE_SPLIT
    selected_features: List[str] = field(default_factory=list)
    eliminated_features: List[str] = field(default_factory=list)
    selection_method: str = ""
    information_leaked_bits: float = 0.0
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "description": self.description,
            "severity": self.severity.value,
            "source_nodes": list(self.source_nodes),
            "affected_features": list(self.affected_features),
            "violation_type": self.violation_type.name,
            "selected_features": list(self.selected_features),
            "eliminated_features": list(self.eliminated_features),
            "selection_method": self.selection_method,
            "information_leaked_bits": self.information_leaked_bits,
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureLeakagePattern":
        """Reconstruct from a plain dictionary produced by :meth:`to_dict`."""
        return cls(
            description=str(data["description"]),
            severity=Severity(data["severity"]),
            source_nodes=list(data.get("source_nodes", [])),
            affected_features=list(data.get("affected_features", [])),
            violation_type=FeatureViolationType[data.get(
                "violation_type", "SELECTION_BEFORE_SPLIT"
            )],
            selected_features=list(data.get("selected_features", [])),
            eliminated_features=list(data.get("eliminated_features", [])),
            selection_method=str(data.get("selection_method", "")),
            information_leaked_bits=float(data.get("information_leaked_bits", 0.0)),
            remediation=str(data.get("remediation", "")),
        )


# ===================================================================
#  Shared helpers
# ===================================================================


def _collect_split_ids(dag: PIDAG) -> Set[str]:
    """Return the set of node IDs that represent a train/test split."""
    split_ops: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })
    return {
        n.node_id for n in dag.nodes.values()
        if n.op_type in split_ops
    }


def _is_before_any_split(
    dag: PIDAG, node_id: str, split_ids: Set[str]
) -> bool:
    """Return *True* if *node_id* is an ancestor of any split node."""
    descendants = dag.descendants(node_id)
    return bool(descendants & split_ids)


def _affected_columns(node: PipelineNode) -> List[str]:
    """Determine which columns a node affects."""
    target_cols = node.metadata.get("columns", None)
    if target_cols and isinstance(target_cols, (list, set, frozenset)):
        return sorted(target_cols)
    return sorted(node.output_columns)


def _selected_columns(node: PipelineNode) -> List[str]:
    """Return the features retained by a selection node."""
    sel = node.metadata.get("selected_features", None)
    if sel and isinstance(sel, (list, set, frozenset)):
        return sorted(sel)
    # Fallback: output columns that are not in the input
    selected = node.metadata.get("support", None)
    if selected and isinstance(selected, (list, set, frozenset)):
        return sorted(selected)
    return sorted(node.output_columns)


def _eliminated_columns(node: PipelineNode) -> List[str]:
    """Return the features eliminated by a selection node."""
    eliminated = node.metadata.get("eliminated_features", None)
    if eliminated and isinstance(eliminated, (list, set, frozenset)):
        return sorted(eliminated)
    # Infer from input/output difference
    removed = node.input_columns - node.output_columns
    return sorted(removed)


def _estimate_leaked_bits(
    node: PipelineNode, n_total_features: int, test_fraction: float
) -> float:
    """Upper-bound the information leaked by a selection decision.

    When a selector sees *N* features from the full dataset with test
    fraction ρ, the selector's decision carries at most
    ``N * ρ * log₂(N)`` bits of test information because each
    binary keep/drop decision is informed by ρ fraction of test rows.
    """
    if n_total_features <= 0 or test_fraction <= 0.0:
        return 0.0
    log_n = math.log2(max(n_total_features, 2))
    return n_total_features * test_fraction * log_n


def _get_test_fraction(node: PipelineNode) -> float:
    """Extract the test fraction from provenance or metadata."""
    if node.provenance:
        fractions = [
            prov.test_fraction
            for prov in node.provenance.values()
            if hasattr(prov, "test_fraction") and prov.test_fraction > 0
        ]
        if fractions:
            return max(fractions)
    return node.metadata.get("test_fraction", 0.25)


def _extract_estimator_class(node: PipelineNode) -> str:
    """Return the estimator class name from node metadata."""
    for key in ("estimator_class", "class_name", "estimator", "model"):
        val = node.metadata.get(key, "")
        if val:
            return str(val)
    if hasattr(node, "estimator_class"):
        return str(node.estimator_class)
    return ""


# ===================================================================
#  SelectionBeforeSplitDetector
# ===================================================================


class SelectionBeforeSplitDetector:
    """Detect feature selection operations on unsplit data.

    Targets sklearn selectors such as ``SelectKBest``,
    ``VarianceThreshold``, ``SelectFromModel``, ``RFE``, and generic
    ``SELECTION`` operations that appear *before* any train/test split
    in the pipeline DAG.

    Each detected violation is categorised by the specific selection
    method so that the leakage estimate and remediation advice can be
    tailored accordingly.
    """

    # Operations that represent feature selection
    _SELECTION_OPS: frozenset[OpType] = frozenset({
        OpType.SELECTION,
        OpType.FIT_TRANSFORM,
        OpType.FIT,
        OpType.TRANSFORM_SK,
    })

    # Splitting operations
    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    # Keywords in estimator class names that map to violation types
    _SELECTOR_CLASS_MAP: Dict[str, FeatureViolationType] = {
        "selectkbest": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "variancethreshold": FeatureViolationType.VARIANCE_THRESHOLD_ON_FULL,
        "genericunivariateselect": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "selectpercentile": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "selectfpr": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "selectfdr": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "selectfwe": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "selectfrommodel": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "rfe": FeatureViolationType.RECURSIVE_ELIMINATION_ON_FULL,
        "rfecv": FeatureViolationType.RECURSIVE_ELIMINATION_ON_FULL,
        "sequentialfeatureselector": FeatureViolationType.STEPWISE_ON_FULL,
    }

    # Score function keywords that refine the violation type
    _SCORE_FUNC_MAP: Dict[str, FeatureViolationType] = {
        "mutual_info": FeatureViolationType.MUTUAL_INFO_ON_FULL,
        "chi2": FeatureViolationType.CHI2_ON_FULL,
        "f_classif": FeatureViolationType.SELECTION_BEFORE_SPLIT,
        "f_regression": FeatureViolationType.SELECTION_BEFORE_SPLIT,
    }

    def __init__(self, *, strict: bool = True) -> None:
        """
        Parameters
        ----------
        strict : bool
            When *True* (default), also flag ``FIT`` and ``FIT_TRANSFORM``
            nodes whose estimator class matches a known selector even if
            the OpType is not ``SELECTION``.
        """
        self._strict = strict

    def detect(self, dag: PIDAG) -> List[FeatureLeakagePattern]:
        """Scan the DAG for feature selection before any split.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[FeatureLeakagePattern]
            Detected feature-selection leakage patterns.
        """
        patterns: List[FeatureLeakagePattern] = []
        split_ids = _collect_split_ids(dag)
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if not self._is_selection_node(node):
                continue
            if _is_before_any_split(dag, node.node_id, split_ids):
                pattern = self._build_pattern(node)
                patterns.append(pattern)

        return patterns

    def _is_selection_node(self, node: PipelineNode) -> bool:
        """Return *True* if the node represents a feature selection step."""
        if node.op_type == OpType.SELECTION:
            return True

        if not self._strict:
            return False

        # Check estimator class for known selectors
        est_class = _extract_estimator_class(node).lower()
        if not est_class:
            return False

        if node.op_type in (OpType.FIT, OpType.FIT_TRANSFORM, OpType.TRANSFORM_SK):
            for keyword in self._SELECTOR_CLASS_MAP:
                if keyword in est_class:
                    return True

        # Check metadata for selection indicators
        if node.metadata.get("is_selector", False):
            return True
        if node.metadata.get("n_features_selected", None) is not None:
            return True

        return False

    def _classify_violation(self, node: PipelineNode) -> FeatureViolationType:
        """Determine the specific violation type from node metadata."""
        est_class = _extract_estimator_class(node).lower()

        # Check estimator class name against known selectors
        for keyword, vtype in self._SELECTOR_CLASS_MAP.items():
            if keyword in est_class:
                return vtype

        # Refine by score function if available
        score_func = str(node.metadata.get("score_func", "")).lower()
        for keyword, vtype in self._SCORE_FUNC_MAP.items():
            if keyword in score_func:
                return vtype

        return FeatureViolationType.SELECTION_BEFORE_SPLIT

    def _build_pattern(self, node: PipelineNode) -> FeatureLeakagePattern:
        """Construct a :class:`FeatureLeakagePattern` for the given node."""
        violation = self._classify_violation(node)
        est_class = _extract_estimator_class(node) or "UnknownSelector"
        affected = _affected_columns(node)
        selected = _selected_columns(node)
        eliminated = _eliminated_columns(node)

        n_features = len(node.input_columns) if node.input_columns else len(affected)
        test_frac = _get_test_fraction(node)
        leaked_bits = _estimate_leaked_bits(node, n_features, test_frac)

        severity = self._assess_severity(n_features, len(eliminated), test_frac)
        method_name = self._format_method_name(est_class, node)

        return FeatureLeakagePattern(
            description=(
                f"Feature selector '{est_class}' at node "
                f"'{node.node_id}' is fitted on the full dataset before "
                f"the train/test split.  The selection decision uses "
                f"test-set statistics, biasing the chosen feature subset."
            ),
            severity=severity,
            source_nodes=[node.node_id],
            affected_features=affected,
            violation_type=violation,
            selected_features=selected,
            eliminated_features=eliminated,
            selection_method=method_name,
            information_leaked_bits=leaked_bits,
            remediation=(
                f"Move the {est_class} step after the train/test split and "
                f"fit it only on the training partition.  Use a Pipeline or "
                f"ColumnTransformer to ensure the selector sees only "
                f"training data during cross-validation."
            ),
        )

    @staticmethod
    def _assess_severity(
        n_features: int, n_eliminated: int, test_fraction: float
    ) -> Severity:
        """Heuristic severity based on selection aggressiveness.

        Eliminating a large fraction of features on the full dataset is
        more harmful because each keep/drop decision is informed by more
        test rows relative to the total.
        """
        if n_features <= 0:
            return Severity.WARNING
        elimination_ratio = n_eliminated / max(n_features, 1)
        if elimination_ratio >= 0.5 or test_fraction >= 0.3:
            return Severity.CRITICAL
        if elimination_ratio >= 0.2 or test_fraction >= 0.15:
            return Severity.WARNING
        return Severity.NEGLIGIBLE

    @staticmethod
    def _format_method_name(est_class: str, node: PipelineNode) -> str:
        """Build a human-readable method name from the estimator class."""
        score_func = node.metadata.get("score_func", "")
        if score_func:
            return f"{est_class}(score_func={score_func})"
        k = node.metadata.get("k", node.metadata.get("n_features_to_select", ""))
        if k:
            return f"{est_class}(k={k})"
        return est_class


# ===================================================================
#  CorrelationFilterAuditor
# ===================================================================


class CorrelationFilterAuditor:
    """Audit correlation-based feature filtering for leakage.

    Correlation filters (Pearson, Spearman, Kendall, mutual information)
    compute pairwise statistics across the dataset.  When these
    statistics are computed on the full (unsplit) data, the resulting
    correlation matrix encodes test-set relationships that influence
    which features are retained or dropped.

    The auditor walks the DAG looking for correlation / covariance
    operations that appear before any split node, as well as metadata
    indicators of correlation-based selection.
    """

    # Operations that compute correlation or covariance matrices
    _CORR_OPS: frozenset[OpType] = frozenset({
        OpType.CORR, OpType.COV,
    })

    # Broader set that includes generic aggregations that may be used
    # for correlation-based selection
    _AGG_OPS: frozenset[OpType] = frozenset({
        OpType.AGG, OpType.AGGREGATE, OpType.DESCRIBE,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    # Keywords that indicate correlation-based filtering in metadata
    _CORR_KEYWORDS: frozenset[str] = frozenset({
        "correlation", "corr", "pearson", "spearman", "kendall",
        "mutual_info", "mutual_information", "mi_score",
        "variance_inflation", "vif", "multicollinearity",
    })

    def __init__(self, *, correlation_threshold: float = 0.0) -> None:
        """
        Parameters
        ----------
        correlation_threshold : float
            If non-zero, only flag correlation computations whose
            ``threshold`` metadata exceeds this value.  When zero
            (default), every pre-split correlation computation is flagged.
        """
        self._correlation_threshold = correlation_threshold

    def audit(self, dag: PIDAG) -> List[FeatureLeakagePattern]:
        """Scan the DAG for correlation-based filtering on unsplit data.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[FeatureLeakagePattern]
            Detected correlation-filter leakage patterns.
        """
        patterns: List[FeatureLeakagePattern] = []
        split_ids = _collect_split_ids(dag)
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if not self._is_correlation_node(node):
                continue
            if _is_before_any_split(dag, node.node_id, split_ids):
                pattern = self._build_correlation_pattern(dag, node)
                if pattern is not None:
                    patterns.append(pattern)

        # Also detect indirect correlation-based selection via downstream
        # drop operations that reference correlation results.
        patterns.extend(self._detect_corr_driven_drops(dag, split_ids))

        return patterns

    def _is_correlation_node(self, node: PipelineNode) -> bool:
        """Return *True* if the node computes a correlation matrix."""
        if node.op_type in self._CORR_OPS:
            return True

        # Check metadata for correlation-related keywords
        method = str(node.metadata.get("method", "")).lower()
        if any(kw in method for kw in self._CORR_KEYWORDS):
            return True

        # Check for aggregation nodes with correlation configuration
        if node.op_type in self._AGG_OPS:
            agg_func = str(node.metadata.get("func", "")).lower()
            if any(kw in agg_func for kw in self._CORR_KEYWORDS):
                return True

        return False

    def _classify_correlation_violation(
        self, node: PipelineNode
    ) -> FeatureViolationType:
        """Determine the specific correlation violation type."""
        method = str(node.metadata.get("method", "")).lower()

        if "mutual_info" in method or "mutual_information" in method:
            return FeatureViolationType.MUTUAL_INFO_ON_FULL
        if "chi2" in method or "chi_square" in method:
            return FeatureViolationType.CHI2_ON_FULL

        # Default: generic correlation filter
        return FeatureViolationType.CORRELATION_FILTER_ON_FULL

    def _build_correlation_pattern(
        self, dag: PIDAG, node: PipelineNode
    ) -> Optional[FeatureLeakagePattern]:
        """Build a leakage pattern for a correlation computation node."""
        # Apply threshold filter if configured
        threshold = node.metadata.get("threshold", 0.0)
        if self._correlation_threshold > 0.0:
            try:
                if float(threshold) < self._correlation_threshold:
                    return None
            except (TypeError, ValueError):
                pass

        violation = self._classify_correlation_violation(node)
        affected = _affected_columns(node)
        method = node.metadata.get("method", "pearson")
        n_features = len(affected)
        test_frac = _get_test_fraction(node)
        leaked_bits = self._estimate_correlation_leakage(
            n_features, test_frac
        )

        # Identify which features the correlation analysis targets
        selected = self._find_corr_selected_features(node)
        eliminated = self._find_corr_eliminated_features(node)

        return FeatureLeakagePattern(
            description=(
                f"Correlation analysis (method='{method}') at node "
                f"'{node.node_id}' is computed on the full dataset "
                f"before the train/test split.  The resulting "
                f"correlation matrix includes test-set relationships."
            ),
            severity=Severity.CRITICAL if n_features >= 10 else Severity.WARNING,
            source_nodes=[node.node_id],
            affected_features=affected,
            violation_type=violation,
            selected_features=selected,
            eliminated_features=eliminated,
            selection_method=f"CorrelationFilter(method={method})",
            information_leaked_bits=leaked_bits,
            remediation=(
                f"Compute the correlation matrix only on the training "
                f"partition after the split.  Drop highly correlated "
                f"features based on training-set correlations alone."
            ),
        )

    def _detect_corr_driven_drops(
        self, dag: PIDAG, split_ids: Set[str]
    ) -> List[FeatureLeakagePattern]:
        """Detect drop operations that appear to be driven by pre-split
        correlation analysis.

        Heuristic: a ``DROP`` node whose metadata references correlation
        (e.g. ``drop_reason='high_correlation'``) that occurs after a
        ``CORR`` node but before the split.
        """
        patterns: List[FeatureLeakagePattern] = []
        drop_ops: frozenset[OpType] = frozenset({OpType.DROP})

        for node in dag.nodes.values():
            if node.op_type not in drop_ops:
                continue
            if not _is_before_any_split(dag, node.node_id, split_ids):
                continue

            drop_reason = str(node.metadata.get("drop_reason", "")).lower()
            if not any(kw in drop_reason for kw in self._CORR_KEYWORDS):
                continue

            # Verify there is a correlation ancestor
            ancestors = dag.ancestors(node.node_id)
            has_corr_ancestor = False
            for anc_id in ancestors:
                anc = dag.get_node(anc_id)
                if self._is_correlation_node(anc):
                    has_corr_ancestor = True
                    break

            if not has_corr_ancestor:
                continue

            eliminated = _eliminated_columns(node)
            affected = _affected_columns(node)

            patterns.append(FeatureLeakagePattern(
                description=(
                    f"Drop node '{node.node_id}' removes features based on "
                    f"correlation analysis performed on unsplit data "
                    f"(reason: '{drop_reason}')."
                ),
                severity=Severity.WARNING,
                source_nodes=[node.node_id],
                affected_features=affected,
                violation_type=FeatureViolationType.CORRELATION_FILTER_ON_FULL,
                eliminated_features=eliminated,
                selection_method=f"CorrelationDrivenDrop(reason={drop_reason})",
                information_leaked_bits=0.0,
                remediation=(
                    "Perform correlation-based feature removal after the "
                    "split, using only training-set correlations."
                ),
            ))

        return patterns

    @staticmethod
    def _estimate_correlation_leakage(
        n_features: int, test_fraction: float
    ) -> float:
        """Estimate information leaked through a correlation matrix.

        A full correlation matrix of *n* features has n*(n-1)/2 unique
        entries.  Each entry is influenced by the test fraction ρ of
        rows, leaking at most ``n*(n-1)/2 * ρ`` bits of information
        (one bit per pairwise relationship being distorted).
        """
        if n_features <= 1 or test_fraction <= 0.0:
            return 0.0
        n_pairs = n_features * (n_features - 1) / 2.0
        return n_pairs * test_fraction

    @staticmethod
    def _find_corr_selected_features(node: PipelineNode) -> List[str]:
        """Return features kept after correlation filtering."""
        sel = node.metadata.get("selected_features", None)
        if sel and isinstance(sel, (list, set, frozenset)):
            return sorted(sel)
        return sorted(node.output_columns)

    @staticmethod
    def _find_corr_eliminated_features(node: PipelineNode) -> List[str]:
        """Return features dropped by correlation filtering."""
        elim = node.metadata.get("eliminated_features", None)
        if elim and isinstance(elim, (list, set, frozenset)):
            return sorted(elim)
        return sorted(node.input_columns - node.output_columns)


# ===================================================================
#  FeatureImportanceLeakDetector
# ===================================================================


class FeatureImportanceLeakDetector:
    """Detect feature importance calculations on the full dataset.

    Tree-based models (random forest, gradient boosting), LASSO, and
    other embedded methods compute feature importance as a by-product
    of fitting.  When this fitting is done on unsplit data, the
    importance scores reflect test-set patterns, and any downstream
    selection based on those scores leaks information.

    Also detects Boruta, recursive feature elimination (RFE), and
    LASSO-based selection when performed before the split.
    """

    # Operations that fit an estimator (importance is a by-product)
    _FIT_OPS: frozenset[OpType] = frozenset({
        OpType.FIT, OpType.FIT_TRANSFORM, OpType.FIT_PREDICT,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    # Estimator class keywords → violation type
    _IMPORTANCE_CLASS_MAP: Dict[str, FeatureViolationType] = {
        "randomforest": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "gradientboosting": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "xgboost": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "xgbclassifier": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "xgbregressor": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "lightgbm": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "lgbmclassifier": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "lgbmregressor": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "catboost": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "extratrees": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "decisiontree": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "lasso": FeatureViolationType.LASSO_SELECTION_ON_FULL,
        "lassocv": FeatureViolationType.LASSO_SELECTION_ON_FULL,
        "elasticnet": FeatureViolationType.LASSO_SELECTION_ON_FULL,
        "elasticnetcv": FeatureViolationType.LASSO_SELECTION_ON_FULL,
        "boruta": FeatureViolationType.BORUTA_ON_FULL,
        "borutapy": FeatureViolationType.BORUTA_ON_FULL,
        "rfe": FeatureViolationType.RECURSIVE_ELIMINATION_ON_FULL,
        "rfecv": FeatureViolationType.RECURSIVE_ELIMINATION_ON_FULL,
        "selectfrommodel": FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
        "sequentialfeatureselector": FeatureViolationType.STEPWISE_ON_FULL,
    }

    # Metadata keys that indicate importance-based selection
    _IMPORTANCE_INDICATORS: frozenset[str] = frozenset({
        "feature_importances_", "feature_importances",
        "coef_", "ranking_", "support_",
        "importance_type", "importance_scores",
    })

    def __init__(self, *, check_importance_metadata: bool = True) -> None:
        """
        Parameters
        ----------
        check_importance_metadata : bool
            When *True* (default), also flag nodes whose metadata
            contains feature-importance arrays even if the estimator
            class is not in the known map.
        """
        self._check_importance_metadata = check_importance_metadata

    def detect(self, dag: PIDAG) -> List[FeatureLeakagePattern]:
        """Scan the DAG for importance-based selection on unsplit data.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[FeatureLeakagePattern]
            Detected feature-importance leakage patterns.
        """
        patterns: List[FeatureLeakagePattern] = []
        split_ids = _collect_split_ids(dag)
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if not self._is_importance_node(node):
                continue
            if _is_before_any_split(dag, node.node_id, split_ids):
                pattern = self._build_importance_pattern(dag, node)
                patterns.append(pattern)

        # Detect selection-from-model chains: a fit node produces
        # importances that feed into a selection node, both before split.
        patterns.extend(self._detect_selection_chains(dag, split_ids))

        return patterns

    def _is_importance_node(self, node: PipelineNode) -> bool:
        """Return *True* if the node produces feature importance scores."""
        if node.op_type not in self._FIT_OPS:
            return False

        est_class = _extract_estimator_class(node).lower()
        for keyword in self._IMPORTANCE_CLASS_MAP:
            if keyword in est_class:
                return True

        if self._check_importance_metadata:
            for indicator in self._IMPORTANCE_INDICATORS:
                if indicator in node.metadata:
                    return True

        return False

    def _classify_importance_violation(
        self, node: PipelineNode
    ) -> FeatureViolationType:
        """Map the estimator to a violation type."""
        est_class = _extract_estimator_class(node).lower()
        for keyword, vtype in self._IMPORTANCE_CLASS_MAP.items():
            if keyword in est_class:
                return vtype
        return FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL

    def _build_importance_pattern(
        self, dag: PIDAG, node: PipelineNode
    ) -> FeatureLeakagePattern:
        """Construct a pattern for an importance-producing node."""
        violation = self._classify_importance_violation(node)
        est_class = _extract_estimator_class(node) or "UnknownEstimator"
        affected = _affected_columns(node)
        n_features = len(node.input_columns) if node.input_columns else len(affected)
        test_frac = _get_test_fraction(node)
        leaked_bits = self._estimate_importance_leakage(
            n_features, test_frac, est_class
        )

        severity = self._assess_importance_severity(
            est_class, n_features, test_frac
        )

        # Attempt to extract importance-based selected/eliminated features
        selected = self._extract_important_features(node)
        eliminated = self._extract_unimportant_features(node, selected)

        return FeatureLeakagePattern(
            description=(
                f"Estimator '{est_class}' at node '{node.node_id}' is "
                f"fitted on the full dataset before the train/test split. "
                f"Feature importances derived from this fit encode "
                f"test-set patterns."
            ),
            severity=severity,
            source_nodes=[node.node_id],
            affected_features=affected,
            violation_type=violation,
            selected_features=selected,
            eliminated_features=eliminated,
            selection_method=f"FeatureImportance({est_class})",
            information_leaked_bits=leaked_bits,
            remediation=(
                f"Fit '{est_class}' only on the training partition.  If "
                f"using feature importances for selection, wrap the "
                f"estimator in a Pipeline so that fitting and selection "
                f"occur within the cross-validation loop."
            ),
        )

    def _detect_selection_chains(
        self, dag: PIDAG, split_ids: Set[str]
    ) -> List[FeatureLeakagePattern]:
        """Detect fit → selection chains that both precede a split.

        A common pattern is:
          1. Fit a random forest on the full data.
          2. Use ``SelectFromModel`` to keep the important features.
          3. Split the data into train/test.

        Both steps (1) and (2) leak information.
        """
        patterns: List[FeatureLeakagePattern] = []
        selection_ops = frozenset({OpType.SELECTION, OpType.FIT_TRANSFORM})

        for node in dag.nodes.values():
            if node.op_type not in selection_ops:
                continue
            if not _is_before_any_split(dag, node.node_id, split_ids):
                continue

            # Check if this selection node has an importance-producing
            # ancestor also before the split.
            ancestor_ids = dag.ancestors(node.node_id)
            importance_ancestors: List[str] = []
            for anc_id in ancestor_ids:
                anc = dag.get_node(anc_id)
                if self._is_importance_node(anc):
                    if _is_before_any_split(dag, anc_id, split_ids):
                        importance_ancestors.append(anc_id)

            if not importance_ancestors:
                continue

            est_class = _extract_estimator_class(node) or "UnknownSelector"
            affected = _affected_columns(node)
            chain_nodes = importance_ancestors + [node.node_id]

            patterns.append(FeatureLeakagePattern(
                description=(
                    f"Selection node '{node.node_id}' ({est_class}) "
                    f"depends on feature importances from "
                    f"{importance_ancestors} which were also fitted on "
                    f"unsplit data.  This forms a two-stage leakage chain."
                ),
                severity=Severity.CRITICAL,
                source_nodes=chain_nodes,
                affected_features=affected,
                violation_type=FeatureViolationType.FEATURE_IMPORTANCE_ON_FULL,
                selection_method=f"ImportanceChain({est_class})",
                information_leaked_bits=0.0,
                remediation=(
                    "Move both the importance estimation and the feature "
                    "selection into the training pipeline, fitting only "
                    "on training data."
                ),
            ))

        return patterns

    def _estimate_importance_leakage(
        self, n_features: int, test_fraction: float, est_class: str
    ) -> float:
        """Estimate bits leaked through importance-based selection.

        Tree-based models learn splits informed by the test rows; the
        resulting importances carry at most ``n_features * ρ * log₂(n)``
        bits.  For LASSO, the coefficient path is influenced by the
        regularisation over the full covariance, leaking a similar
        amount.
        """
        if n_features <= 0 or test_fraction <= 0.0:
            return 0.0
        log_n = math.log2(max(n_features, 2))
        base_leakage = n_features * test_fraction * log_n

        # Tree-based models have higher capacity than linear models
        est_lower = est_class.lower()
        if any(kw in est_lower for kw in (
            "randomforest", "gradientboosting", "xgb", "lightgbm",
            "catboost", "extratrees",
        )):
            return base_leakage * 1.5
        if any(kw in est_lower for kw in ("lasso", "elasticnet")):
            return base_leakage * 0.8
        return base_leakage

    @staticmethod
    def _assess_importance_severity(
        est_class: str, n_features: int, test_fraction: float
    ) -> Severity:
        """Heuristic severity for importance-based leakage."""
        est_lower = est_class.lower()
        # Tree ensembles on many features are high-capacity learners
        is_tree_ensemble = any(kw in est_lower for kw in (
            "randomforest", "gradientboosting", "xgb", "lightgbm",
            "catboost", "extratrees",
        ))
        if is_tree_ensemble and n_features >= 20:
            return Severity.CRITICAL
        if is_tree_ensemble or n_features >= 50:
            return Severity.CRITICAL
        if n_features >= 10 or test_fraction >= 0.3:
            return Severity.WARNING
        return Severity.NEGLIGIBLE

    @staticmethod
    def _extract_important_features(node: PipelineNode) -> List[str]:
        """Return features deemed important from node metadata."""
        for key in ("selected_features", "support_", "important_features"):
            val = node.metadata.get(key, None)
            if val and isinstance(val, (list, set, frozenset)):
                return sorted(val)
        return sorted(node.output_columns)

    @staticmethod
    def _extract_unimportant_features(
        node: PipelineNode, selected: List[str]
    ) -> List[str]:
        """Return features deemed unimportant (excluded by importance)."""
        for key in ("eliminated_features", "unimportant_features"):
            val = node.metadata.get(key, None)
            if val and isinstance(val, (list, set, frozenset)):
                return sorted(val)
        selected_set = frozenset(selected)
        return sorted(node.input_columns - selected_set)


# ===================================================================
#  DimensionalityReductionAuditor
# ===================================================================


class DimensionalityReductionAuditor:
    """Audit dimensionality reduction for leakage.

    PCA, truncated SVD, UMAP, t-SNE, kernel PCA, and similar
    techniques learn a projection from the input feature space to a
    lower-dimensional space.  When this projection is learned on the
    full (unsplit) dataset, the principal components or embedding encode
    test-set variance, which then flows into the training set.

    The auditor identifies fitting of known dimensionality-reduction
    estimators that occur before any train/test split in the DAG.
    """

    _FIT_OPS: frozenset[OpType] = frozenset({
        OpType.FIT, OpType.FIT_TRANSFORM, OpType.FIT_PREDICT,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    # Known dimensionality-reduction estimator class keywords
    _DIM_REDUCTION_CLASSES: frozenset[str] = frozenset({
        "pca", "truncatedsvd", "incrementalpca", "sparsepca",
        "minibatchsparsepca", "kernelpca", "factoranalysis",
        "fastica", "nmf", "latentdirichletallocation", "lda",
        "tsne", "umap", "isomap", "locallylinearembedding",
        "lle", "mds", "spectralembedding",
        "sparserandomprojection", "gaussianrandomprojection",
        "decomposition", "manifold",
    })

    # Metadata keys that indicate dimensionality reduction
    _DIM_REDUCTION_INDICATORS: frozenset[str] = frozenset({
        "n_components", "explained_variance_", "explained_variance_ratio_",
        "components_", "singular_values_", "embedding_",
    })

    def __init__(self, *, min_components: int = 0) -> None:
        """
        Parameters
        ----------
        min_components : int
            Only flag dimensionality reduction if the number of output
            components is at least this value.  0 (default) flags all.
        """
        self._min_components = min_components

    def audit(self, dag: PIDAG) -> List[FeatureLeakagePattern]:
        """Scan the DAG for dimensionality reduction on unsplit data.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[FeatureLeakagePattern]
            Detected dimensionality-reduction leakage patterns.
        """
        patterns: List[FeatureLeakagePattern] = []
        split_ids = _collect_split_ids(dag)
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if not self._is_dim_reduction_node(node):
                continue
            if _is_before_any_split(dag, node.node_id, split_ids):
                pattern = self._build_dim_reduction_pattern(dag, node)
                if pattern is not None:
                    patterns.append(pattern)

        return patterns

    def _is_dim_reduction_node(self, node: PipelineNode) -> bool:
        """Return *True* if the node fits a dimensionality-reduction model."""
        if node.op_type not in self._FIT_OPS:
            return False

        est_class = _extract_estimator_class(node).lower()
        for keyword in self._DIM_REDUCTION_CLASSES:
            if keyword in est_class:
                return True

        # Check metadata for dimensionality-reduction indicators
        for indicator in self._DIM_REDUCTION_INDICATORS:
            if indicator in node.metadata:
                return True

        # Check if technique metadata indicates DR
        technique = str(node.metadata.get("technique", "")).lower()
        if any(kw in technique for kw in self._DIM_REDUCTION_CLASSES):
            return True

        return False

    def _classify_dim_reduction_violation(
        self, node: PipelineNode
    ) -> FeatureViolationType:
        """Determine the specific violation type for DR."""
        est_class = _extract_estimator_class(node).lower()

        if "pca" in est_class or "principalcomponent" in est_class:
            return FeatureViolationType.PCA_ON_FULL
        if "svd" in est_class:
            return FeatureViolationType.PCA_ON_FULL
        if any(kw in est_class for kw in ("umap", "tsne", "isomap")):
            return FeatureViolationType.PCA_ON_FULL

        # Default
        return FeatureViolationType.PCA_ON_FULL

    def _build_dim_reduction_pattern(
        self, dag: PIDAG, node: PipelineNode
    ) -> Optional[FeatureLeakagePattern]:
        """Construct a pattern for a dimensionality-reduction node."""
        n_components = self._get_n_components(node)
        if self._min_components > 0 and n_components < self._min_components:
            return None

        violation = self._classify_dim_reduction_violation(node)
        est_class = _extract_estimator_class(node) or "UnknownDimReducer"
        affected = _affected_columns(node)
        n_input_features = len(node.input_columns) if node.input_columns else len(affected)
        test_frac = _get_test_fraction(node)

        leaked_bits = self._estimate_dr_leakage(
            n_input_features, n_components, test_frac
        )
        severity = self._assess_dr_severity(
            n_input_features, n_components, test_frac
        )

        explained_var = self._get_explained_variance(node)
        components_desc = (
            f"n_components={n_components}" if n_components > 0
            else "n_components=unknown"
        )
        var_desc = (
            f", explained_variance={explained_var:.2%}"
            if explained_var > 0.0 else ""
        )

        return FeatureLeakagePattern(
            description=(
                f"Dimensionality reduction '{est_class}' "
                f"({components_desc}{var_desc}) at node "
                f"'{node.node_id}' is fitted on the full dataset before "
                f"the train/test split.  The learned projection encodes "
                f"test-set variance structure."
            ),
            severity=severity,
            source_nodes=[node.node_id],
            affected_features=affected,
            violation_type=violation,
            selected_features=self._get_component_names(node, n_components),
            eliminated_features=self._get_discarded_features(node, n_components),
            selection_method=f"DimensionalityReduction({est_class}, {components_desc})",
            information_leaked_bits=leaked_bits,
            remediation=(
                f"Fit '{est_class}' only on the training partition after "
                f"the split.  Use ``Pipeline`` to ensure the "
                f"dimensionality reduction step is refitted for each "
                f"cross-validation fold."
            ),
        )

    @staticmethod
    def _get_n_components(node: PipelineNode) -> int:
        """Extract the number of components from metadata."""
        for key in ("n_components", "n_components_", "k"):
            val = node.metadata.get(key, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        # Infer from output schema
        n_out = len(node.output_columns)
        n_in = len(node.input_columns)
        if n_out < n_in and n_out > 0:
            return n_out
        return 0

    @staticmethod
    def _get_explained_variance(node: PipelineNode) -> float:
        """Extract the total explained variance ratio if available."""
        evr = node.metadata.get("explained_variance_ratio_", None)
        if evr is not None:
            try:
                if isinstance(evr, (list, tuple)):
                    return float(sum(evr))
                return float(evr)
            except (TypeError, ValueError):
                pass
        return 0.0

    @staticmethod
    def _estimate_dr_leakage(
        n_input_features: int, n_components: int, test_fraction: float
    ) -> float:
        """Estimate bits leaked by dimensionality reduction.

        PCA computes the covariance matrix (n×n) and extracts
        eigenvectors.  Each eigenvector is influenced by the test
        fraction ρ of rows.  The leakage is bounded by the information
        in the covariance entries: ``n*(n-1)/2 * ρ`` bits for the
        covariance matrix, attenuated by the fraction of variance
        retained (n_components/n_input_features).
        """
        if n_input_features <= 1 or test_fraction <= 0.0:
            return 0.0
        n_pairs = n_input_features * (n_input_features - 1) / 2.0
        retention_ratio = (
            n_components / max(n_input_features, 1)
            if n_components > 0 else 1.0
        )
        return n_pairs * test_fraction * retention_ratio

    @staticmethod
    def _assess_dr_severity(
        n_input_features: int, n_components: int, test_fraction: float
    ) -> Severity:
        """Heuristic severity for dimensionality reduction leakage."""
        if n_input_features >= 50 or test_fraction >= 0.3:
            return Severity.CRITICAL
        if n_input_features >= 10:
            return Severity.WARNING
        if n_components > 0 and n_components < n_input_features // 2:
            return Severity.WARNING
        return Severity.NEGLIGIBLE

    @staticmethod
    def _get_component_names(
        node: PipelineNode, n_components: int
    ) -> List[str]:
        """Return names of the generated components."""
        comp_names = node.metadata.get("component_names", None)
        if comp_names and isinstance(comp_names, (list, set, frozenset)):
            return sorted(comp_names)
        if n_components > 0:
            return [f"component_{i}" for i in range(n_components)]
        return sorted(node.output_columns)

    @staticmethod
    def _get_discarded_features(
        node: PipelineNode, n_components: int
    ) -> List[str]:
        """Return original features that are no longer directly present
        in the output (compressed into the components)."""
        if n_components <= 0:
            return []
        return sorted(node.input_columns - node.output_columns)


# ===================================================================
#  FeatureLeakageDetector  (top-level orchestrator)
# ===================================================================


class FeatureLeakageDetector:
    """Orchestrator that runs all feature-leakage sub-detectors on a
    PI-DAG and returns the combined list of detected patterns.

    Usage::

        detector = FeatureLeakageDetector()
        patterns = detector.detect(dag)
        for p in patterns:
            print(p.severity, p.description)
    """

    def __init__(
        self,
        *,
        strict: bool = True,
        correlation_threshold: float = 0.0,
        check_importance_metadata: bool = True,
        min_components: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        strict : bool
            Passed to :class:`SelectionBeforeSplitDetector`.
        correlation_threshold : float
            Passed to :class:`CorrelationFilterAuditor`.
        check_importance_metadata : bool
            Passed to :class:`FeatureImportanceLeakDetector`.
        min_components : int
            Passed to :class:`DimensionalityReductionAuditor`.
        """
        self._selection = SelectionBeforeSplitDetector(strict=strict)
        self._correlation = CorrelationFilterAuditor(
            correlation_threshold=correlation_threshold,
        )
        self._importance = FeatureImportanceLeakDetector(
            check_importance_metadata=check_importance_metadata,
        )
        self._dim_reduction = DimensionalityReductionAuditor(
            min_components=min_components,
        )

    # -- public properties ---------------------------------------------------

    @property
    def selection_detector(self) -> SelectionBeforeSplitDetector:
        """Access the underlying :class:`SelectionBeforeSplitDetector`."""
        return self._selection

    @property
    def correlation_auditor(self) -> CorrelationFilterAuditor:
        """Access the underlying :class:`CorrelationFilterAuditor`."""
        return self._correlation

    @property
    def importance_detector(self) -> FeatureImportanceLeakDetector:
        """Access the underlying :class:`FeatureImportanceLeakDetector`."""
        return self._importance

    @property
    def dim_reduction_auditor(self) -> DimensionalityReductionAuditor:
        """Access the underlying :class:`DimensionalityReductionAuditor`."""
        return self._dim_reduction

    # -- detection -----------------------------------------------------------

    def detect(self, dag: PIDAG) -> List[FeatureLeakagePattern]:
        """Run all sub-detectors and return the combined patterns.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[FeatureLeakagePattern]
            Detected feature leakage patterns, ordered by severity
            (critical first).
        """
        patterns: List[FeatureLeakagePattern] = []
        patterns.extend(self._selection.detect(dag))
        patterns.extend(self._correlation.audit(dag))
        patterns.extend(self._importance.detect(dag))
        patterns.extend(self._dim_reduction.audit(dag))

        # Deduplicate patterns that reference the same source node
        patterns = self._deduplicate(patterns)
        patterns.sort(key=lambda p: p.severity, reverse=True)
        return patterns

    def detect_with_summary(
        self, dag: PIDAG
    ) -> Tuple[List[FeatureLeakagePattern], Dict[str, Any]]:
        """Like :meth:`detect` but also returns an aggregated summary.

        Returns
        -------
        tuple[list[FeatureLeakagePattern], dict]
            ``(patterns, summary)`` where *summary* contains:

            * ``n_patterns`` – total number of detected patterns.
            * ``n_critical`` – patterns with CRITICAL severity.
            * ``n_warning`` – patterns with WARNING severity.
            * ``n_negligible`` – patterns with NEGLIGIBLE severity.
            * ``total_leaked_bits`` – sum of estimated leaked bits.
            * ``violation_counts`` – dict mapping violation type name to
              count.
            * ``affected_feature_count`` – number of unique affected
              features across all patterns.
            * ``selection_methods`` – list of distinct selection methods.
        """
        patterns = self.detect(dag)
        summary = self._build_summary(patterns)
        return patterns, summary

    @staticmethod
    def _deduplicate(
        patterns: List[FeatureLeakagePattern],
    ) -> List[FeatureLeakagePattern]:
        """Remove duplicate patterns that share all source nodes.

        When the same node is flagged by multiple sub-detectors, keep
        only the highest-severity instance.
        """
        seen: Dict[Tuple[str, ...], FeatureLeakagePattern] = {}
        for pattern in patterns:
            key = tuple(sorted(pattern.source_nodes))
            existing = seen.get(key)
            if existing is None:
                seen[key] = pattern
            elif pattern.severity.value > existing.severity.value:
                seen[key] = pattern
        return list(seen.values())

    @staticmethod
    def _build_summary(
        patterns: List[FeatureLeakagePattern],
    ) -> Dict[str, Any]:
        """Aggregate statistics across all detected patterns."""
        n_critical = sum(
            1 for p in patterns if p.severity == Severity.CRITICAL
        )
        n_warning = sum(
            1 for p in patterns if p.severity == Severity.WARNING
        )
        n_negligible = sum(
            1 for p in patterns if p.severity == Severity.NEGLIGIBLE
        )
        total_bits = sum(p.information_leaked_bits for p in patterns)

        violation_counts: Dict[str, int] = {}
        for p in patterns:
            name = p.violation_type.name
            violation_counts[name] = violation_counts.get(name, 0) + 1

        affected_features: Set[str] = set()
        for p in patterns:
            affected_features.update(p.affected_features)

        methods: Set[str] = set()
        for p in patterns:
            if p.selection_method:
                methods.add(p.selection_method)

        return {
            "n_patterns": len(patterns),
            "n_critical": n_critical,
            "n_warning": n_warning,
            "n_negligible": n_negligible,
            "total_leaked_bits": total_bits,
            "violation_counts": violation_counts,
            "affected_feature_count": len(affected_features),
            "selection_methods": sorted(methods),
        }
