"""
taintflow.patterns.target_leakage – Target leakage pattern detectors.

Target leakage occurs when information about the target variable
improperly flows into the training features, enabling the model to
"cheat" during training.  Common manifestations include:

* The target column itself (or a renamed copy) is left in the feature
  matrix.
* Features are derived from the target (e.g. ``y - mean(y)``) before
  or after splitting.
* Target encoding (mean encoding, frequency encoding, etc.) is fitted
  on the **full** dataset instead of only the training fold.
* Proxy features that are near-perfect correlates of the target (e.g.
  an approval flag that mirrors the label) remain in the feature set.
* Aggregations (groupby-mean, pivot-table) that incorporate the target
  column are used to generate features before splitting.

The entry-point is :class:`TargetLeakageDetector`, which orchestrates
the specialised sub-detectors and returns a list of
:class:`TargetLeakagePattern` instances.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

from taintflow.core.types import OpType, Origin, Severity

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG
    from taintflow.dag.node import PipelineNode
    from taintflow.dag.edge import PipelineEdge


# ===================================================================
#  Supporting enumerations
# ===================================================================


class TargetViolationType(Enum):
    """Specific kind of target leakage violation."""

    DIRECT_TARGET_INCLUSION = auto()
    TARGET_DERIVED_FEATURE = auto()
    TARGET_ENCODING_LEAKAGE = auto()
    PROXY_FEATURE_CORRELATION = auto()
    TARGET_AGGREGATION_LEAKAGE = auto()
    LABEL_ENCODING_ON_TARGET = auto()
    FUTURE_TARGET_USAGE = auto()
    TARGET_IN_JOIN_KEY = auto()
    TARGET_COPY_ALIAS = auto()
    TARGET_STATISTICAL_LEAK = auto()
    TARGET_IN_GROUPBY = auto()


# ===================================================================
#  Pattern dataclass
# ===================================================================


@dataclass
class TargetLeakagePattern:
    """A detected instance of target leakage in the pipeline DAG.

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
    violation_type : TargetViolationType
        Classification of the violation.
    target_column : str
        Name of the target column involved in the leak.
    feature_path : list[str]
        Ordered node IDs tracing the information flow from the target
        to the contaminated feature(s).
    correlation_score : float
        Estimated Pearson / rank correlation between the leaked feature
        and the target (0.0 when unknown).
    mutual_information_bits : float
        Estimated mutual information in bits between the leaked feature
        and the target (0.0 when unknown).
    remediation : str
        Suggested fix for the detected leakage.
    """

    description: str
    severity: Severity
    source_nodes: List[str]
    affected_features: List[str]
    violation_type: TargetViolationType = TargetViolationType.DIRECT_TARGET_INCLUSION
    target_column: str = ""
    feature_path: List[str] = field(default_factory=list)
    correlation_score: float = 0.0
    mutual_information_bits: float = 0.0
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "description": self.description,
            "severity": self.severity.value,
            "source_nodes": list(self.source_nodes),
            "affected_features": list(self.affected_features),
            "violation_type": self.violation_type.name,
            "target_column": self.target_column,
            "feature_path": list(self.feature_path),
            "correlation_score": self.correlation_score,
            "mutual_information_bits": self.mutual_information_bits,
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetLeakagePattern":
        """Deserialise from a plain dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict` (or equivalent).

        Returns
        -------
        TargetLeakagePattern
        """
        severity_raw = data.get("severity", "warning")
        if isinstance(severity_raw, str):
            severity = Severity(severity_raw)
        else:
            severity = severity_raw

        violation_raw = data.get("violation_type", "DIRECT_TARGET_INCLUSION")
        if isinstance(violation_raw, str):
            violation = TargetViolationType[violation_raw]
        else:
            violation = violation_raw

        return cls(
            description=data.get("description", ""),
            severity=severity,
            source_nodes=list(data.get("source_nodes", [])),
            affected_features=list(data.get("affected_features", [])),
            violation_type=violation,
            target_column=data.get("target_column", ""),
            feature_path=list(data.get("feature_path", [])),
            correlation_score=float(data.get("correlation_score", 0.0)),
            mutual_information_bits=float(data.get("mutual_information_bits", 0.0)),
            remediation=data.get("remediation", ""),
        )


# ===================================================================
#  TargetLeakageSeverityEstimator
# ===================================================================


class TargetLeakageSeverityEstimator:
    """Estimate the severity of target leakage from information-theoretic
    and statistical bounds.

    The estimator maps raw measurements (mutual information in bits,
    correlation coefficients, and violation type metadata) into the
    three-level :class:`Severity` scale used throughout TaintFlow.

    Parameters
    ----------
    mi_warn_threshold : float
        Mutual-information threshold (bits) at or above which the
        severity is at least ``WARNING``.  Default ``0.5``.
    mi_critical_threshold : float
        Mutual-information threshold (bits) at or above which the
        severity is ``CRITICAL``.  Default ``4.0``.
    corr_warn_threshold : float
        Absolute correlation threshold for ``WARNING``.  Default ``0.7``.
    corr_critical_threshold : float
        Absolute correlation threshold for ``CRITICAL``.  Default ``0.95``.
    """

    def __init__(
        self,
        *,
        mi_warn_threshold: float = 0.5,
        mi_critical_threshold: float = 4.0,
        corr_warn_threshold: float = 0.7,
        corr_critical_threshold: float = 0.95,
    ) -> None:
        self._mi_warn = mi_warn_threshold
        self._mi_crit = mi_critical_threshold
        self._corr_warn = corr_warn_threshold
        self._corr_crit = corr_critical_threshold

    # ----- public API ---------------------------------------------------

    def estimate(
        self,
        *,
        mutual_information_bits: float = 0.0,
        correlation_score: float = 0.0,
        violation_type: Optional[TargetViolationType] = None,
    ) -> Severity:
        """Return the most severe rating implied by the given evidence.

        The final severity is the *maximum* of three independent
        assessments: MI-based, correlation-based, and violation-type
        floor.

        Parameters
        ----------
        mutual_information_bits : float
            Estimated MI in bits.
        correlation_score : float
            Absolute Pearson/Spearman correlation.
        violation_type : TargetViolationType or None
            If provided, applies a type-specific severity floor.

        Returns
        -------
        Severity
        """
        candidates: List[Severity] = [
            self.severity_from_mi(mutual_information_bits),
            self.severity_from_correlation(correlation_score),
        ]
        if violation_type is not None:
            candidates.append(self._violation_floor(violation_type))
        return max(candidates)

    def severity_from_mi(self, bits: float) -> Severity:
        """Map mutual information (bits) to a severity level.

        Parameters
        ----------
        bits : float
            Non-negative MI estimate.

        Returns
        -------
        Severity
        """
        if bits >= self._mi_crit:
            return Severity.CRITICAL
        if bits >= self._mi_warn:
            return Severity.WARNING
        return Severity.NEGLIGIBLE

    def severity_from_correlation(self, corr: float) -> Severity:
        """Map an absolute correlation coefficient to a severity level.

        Parameters
        ----------
        corr : float
            Absolute Pearson or Spearman correlation (0–1).

        Returns
        -------
        Severity
        """
        abs_corr = abs(corr)
        if abs_corr >= self._corr_crit:
            return Severity.CRITICAL
        if abs_corr >= self._corr_warn:
            return Severity.WARNING
        return Severity.NEGLIGIBLE

    def mi_upper_bound_binary(self, accuracy: float) -> float:
        """Compute an upper bound on MI for a binary target given
        downstream classifier accuracy.

        Uses the identity  MI ≤ 1 − H(error)  where
        H(p) = −p log₂ p − (1−p) log₂ (1−p)  and error = 1 − accuracy.

        Parameters
        ----------
        accuracy : float
            Classifier accuracy on a held-out fold (0–1).

        Returns
        -------
        float
            Upper bound on MI in bits.
        """
        error = 1.0 - max(0.0, min(1.0, accuracy))
        if error <= 0.0 or error >= 1.0:
            return 1.0
        h_error = -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
        return max(0.0, 1.0 - h_error)

    def mi_upper_bound_multiclass(self, accuracy: float, n_classes: int) -> float:
        """Compute an MI upper bound for a multiclass target.

        Uses  MI ≤ log₂(K) − H(error)  where K is the number of
        classes.

        Parameters
        ----------
        accuracy : float
            Classifier accuracy on a held-out fold (0–1).
        n_classes : int
            Number of target classes (≥ 2).

        Returns
        -------
        float
            Upper bound on MI in bits.
        """
        n_classes = max(2, n_classes)
        error = 1.0 - max(0.0, min(1.0, accuracy))
        if error <= 0.0 or error >= 1.0:
            return math.log2(n_classes)
        h_error = -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
        return max(0.0, math.log2(n_classes) - h_error)

    # ----- internal helpers ---------------------------------------------

    @staticmethod
    def _violation_floor(violation_type: TargetViolationType) -> Severity:
        """Return a minimum severity based on the violation type alone.

        Certain violation types (e.g. direct target inclusion) are
        always critical regardless of the measured MI / correlation.
        """
        always_critical: frozenset[TargetViolationType] = frozenset({
            TargetViolationType.DIRECT_TARGET_INCLUSION,
            TargetViolationType.TARGET_COPY_ALIAS,
        })
        always_warning: frozenset[TargetViolationType] = frozenset({
            TargetViolationType.TARGET_ENCODING_LEAKAGE,
            TargetViolationType.TARGET_AGGREGATION_LEAKAGE,
            TargetViolationType.LABEL_ENCODING_ON_TARGET,
            TargetViolationType.TARGET_IN_GROUPBY,
            TargetViolationType.TARGET_DERIVED_FEATURE,
            TargetViolationType.FUTURE_TARGET_USAGE,
            TargetViolationType.TARGET_IN_JOIN_KEY,
        })
        if violation_type in always_critical:
            return Severity.CRITICAL
        if violation_type in always_warning:
            return Severity.WARNING
        return Severity.NEGLIGIBLE


# ===================================================================
#  DirectTargetDetector
# ===================================================================


class DirectTargetDetector:
    """Detect when the target column is directly included in the feature
    matrix.

    This is the most blatant form of target leakage: the label column
    is never removed from the DataFrame before training.  The detector
    works in two modes:

    1. **Explicit target** – the user annotates a column as the target
       in the DAG metadata; we check whether that column appears in any
       downstream feature-engineering or model-fit node's inputs.
    2. **Heuristic** – when no explicit annotation exists the detector
       looks for columns named ``target``, ``label``, ``y``, or ending
       with ``_target`` / ``_label``.

    Parameters
    ----------
    target_column_names : set[str] or None
        Explicit target column names.  When ``None`` the detector
        falls back to heuristics.
    check_aliases : bool
        If ``True`` (default), also detect renamed copies of the target
        (e.g. ``y_copy``, ``target_backup``).
    """

    # Heuristic target-name patterns
    _TARGET_EXACT: frozenset[str] = frozenset({
        "target", "label", "y", "class", "outcome",
    })
    _TARGET_SUFFIXES: Tuple[str, ...] = (
        "_target", "_label", "_y", "_class", "_outcome",
    )
    _TARGET_PREFIXES: Tuple[str, ...] = (
        "target_", "label_", "is_", "has_",
    )

    # Operations that consume features for modelling
    _FIT_OPS: frozenset[OpType] = frozenset({
        OpType.FIT, OpType.FIT_TRANSFORM, OpType.FIT_PREDICT,
        OpType.ESTIMATOR_FIT,
    })

    # Operations that select / drop columns
    _DROP_OPS: frozenset[OpType] = frozenset({
        OpType.DROP, OpType.GETITEM, OpType.LOC, OpType.ILOC,
    })

    # Splitting operations
    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    def __init__(
        self,
        *,
        target_column_names: Optional[Set[str]] = None,
        check_aliases: bool = True,
    ) -> None:
        self._explicit_targets: Optional[Set[str]] = (
            set(target_column_names) if target_column_names else None
        )
        self._check_aliases = check_aliases

    # ----- public API ---------------------------------------------------

    def detect(self, dag: PIDAG) -> List[TargetLeakagePattern]:
        """Scan the DAG for direct target inclusion in feature inputs.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[TargetLeakagePattern]
        """
        patterns: List[TargetLeakagePattern] = []
        target_cols = self._resolve_targets(dag)
        if not target_cols:
            return patterns

        fit_nodes = [
            n for n in dag.nodes.values()
            if n.op_type in self._FIT_OPS
        ]
        for fit_node in fit_nodes:
            patterns.extend(
                self._check_fit_node(dag, fit_node, target_cols)
            )

        if self._check_aliases:
            patterns.extend(self._detect_aliases(dag, target_cols))

        return patterns

    # ----- internal helpers ---------------------------------------------

    def _resolve_targets(self, dag: PIDAG) -> Set[str]:
        """Determine the set of target column names.

        Prefers explicit configuration, then DAG-level metadata, and
        finally falls back to heuristic name matching.
        """
        if self._explicit_targets:
            return set(self._explicit_targets)

        meta_target = dag.metadata.get("target_column", None)
        if meta_target:
            if isinstance(meta_target, str):
                return {meta_target}
            if isinstance(meta_target, (list, set, frozenset)):
                return set(meta_target)

        return self._heuristic_targets(dag)

    def _heuristic_targets(self, dag: PIDAG) -> Set[str]:
        """Identify probable target columns by name patterns."""
        candidates: Set[str] = set()
        all_columns: Set[str] = set()
        for node in dag.nodes.values():
            all_columns.update(node.output_columns)

        for col in all_columns:
            lower = col.lower()
            if lower in self._TARGET_EXACT:
                candidates.add(col)
                continue
            if any(lower.endswith(sfx) for sfx in self._TARGET_SUFFIXES):
                candidates.add(col)
                continue
            if any(lower.startswith(pfx) for pfx in self._TARGET_PREFIXES):
                candidates.add(col)
        return candidates

    def _check_fit_node(
        self,
        dag: PIDAG,
        fit_node: PipelineNode,
        target_cols: Set[str],
    ) -> List[TargetLeakagePattern]:
        """Check whether *fit_node* receives any target column as a
        feature input (not as the explicit ``y`` argument)."""
        patterns: List[TargetLeakagePattern] = []

        # Columns arriving as feature inputs
        feature_inputs = self._feature_inputs_for(dag, fit_node)
        leaked = feature_inputs & target_cols

        if not leaked:
            return patterns

        # Verify the target was not properly separated upstream
        for col in sorted(leaked):
            if self._is_properly_separated(dag, fit_node, col):
                continue

            path = self._trace_target_path(dag, fit_node, col)
            patterns.append(TargetLeakagePattern(
                description=(
                    f"Target column '{col}' is present in the feature inputs "
                    f"of fit node '{fit_node.node_id}'. This constitutes "
                    f"direct target leakage."
                ),
                severity=Severity.CRITICAL,
                source_nodes=[fit_node.node_id],
                affected_features=[col],
                violation_type=TargetViolationType.DIRECT_TARGET_INCLUSION,
                target_column=col,
                feature_path=path,
                correlation_score=1.0,
                mutual_information_bits=0.0,
                remediation=(
                    f"Drop the target column '{col}' from the feature "
                    f"matrix before fitting the model."
                ),
            ))
        return patterns

    def _feature_inputs_for(
        self, dag: PIDAG, fit_node: PipelineNode
    ) -> Set[str]:
        """Return column names that serve as *feature* inputs to a fit
        node (excluding the ``y`` / label argument)."""
        y_cols: Set[str] = set()
        y_meta = fit_node.metadata.get("y_column", None)
        if y_meta:
            if isinstance(y_meta, str):
                y_cols.add(y_meta)
            elif isinstance(y_meta, (list, set, frozenset)):
                y_cols.update(y_meta)

        label_edge_cols: Set[str] = set()
        for edge in dag.in_edges(fit_node.node_id):
            if hasattr(edge, "kind") and edge.kind.name == "LABEL_FLOW":
                src = dag.get_node(edge.source)
                label_edge_cols.update(src.output_columns)

        excluded = y_cols | label_edge_cols
        return set(fit_node.input_columns) - excluded

    def _is_properly_separated(
        self, dag: PIDAG, fit_node: PipelineNode, target_col: str
    ) -> bool:
        """Return *True* if the target column was removed from features
        upstream of the fit node via a DROP or selection operation."""
        ancestors = dag.ancestors(fit_node.node_id)
        topo = dag.topological_order()
        ordered = [nid for nid in topo if nid in ancestors]

        for nid in reversed(ordered):
            node = dag.get_node(nid)
            if node.op_type == OpType.DROP:
                dropped = node.metadata.get("columns", [])
                if target_col in dropped:
                    return True
            if node.op_type == OpType.GETITEM:
                selected = node.metadata.get("columns", [])
                if selected and target_col not in selected:
                    return True
        return False

    def _trace_target_path(
        self, dag: PIDAG, fit_node: PipelineNode, target_col: str
    ) -> List[str]:
        """Return an ordered list of node IDs tracing how *target_col*
        flows from its origin to *fit_node*."""
        path: List[str] = []
        ancestors = dag.ancestors(fit_node.node_id)
        topo = dag.topological_order()
        ordered = [nid for nid in topo if nid in ancestors]

        for nid in ordered:
            node = dag.get_node(nid)
            if target_col in node.output_columns:
                path.append(nid)
        path.append(fit_node.node_id)
        return path

    def _detect_aliases(
        self, dag: PIDAG, target_cols: Set[str]
    ) -> List[TargetLeakagePattern]:
        """Detect renamed copies of the target column in feature inputs.

        A column is suspected to be an alias if it was produced by a
        RENAME or ASSIGN node whose input references a known target
        column, or if its provenance traces directly back to a target.
        """
        patterns: List[TargetLeakagePattern] = []
        alias_map: Dict[str, str] = {}

        for node in dag.nodes.values():
            if node.op_type == OpType.RENAME:
                mapping = node.metadata.get("columns", {})
                if isinstance(mapping, dict):
                    for old_name, new_name in mapping.items():
                        if old_name in target_cols:
                            alias_map[new_name] = old_name
            if node.op_type == OpType.ASSIGN:
                expr = node.metadata.get("expression", "")
                for tc in target_cols:
                    if tc in str(expr):
                        new_cols = set(node.output_columns) - set(node.input_columns)
                        for nc in new_cols:
                            alias_map[nc] = tc

        if not alias_map:
            return patterns

        fit_nodes = [
            n for n in dag.nodes.values()
            if n.op_type in self._FIT_OPS
        ]
        for fit_node in fit_nodes:
            feature_inputs = self._feature_inputs_for(dag, fit_node)
            for alias, original in alias_map.items():
                if alias in feature_inputs:
                    path = self._trace_target_path(dag, fit_node, alias)
                    patterns.append(TargetLeakagePattern(
                        description=(
                            f"Column '{alias}' in fit node "
                            f"'{fit_node.node_id}' is an alias of target "
                            f"column '{original}'. This constitutes direct "
                            f"target leakage via renaming."
                        ),
                        severity=Severity.CRITICAL,
                        source_nodes=[fit_node.node_id],
                        affected_features=[alias],
                        violation_type=TargetViolationType.TARGET_COPY_ALIAS,
                        target_column=original,
                        feature_path=path,
                        correlation_score=1.0,
                        mutual_information_bits=0.0,
                        remediation=(
                            f"Drop the alias column '{alias}' (copy of "
                            f"target '{original}') from the feature matrix."
                        ),
                    ))
        return patterns


# ===================================================================
#  TargetEncodingAuditor
# ===================================================================


class TargetEncodingAuditor:
    """Audit target encoding operations for leakage.

    Target encoding (also known as mean encoding, likelihood encoding,
    or impact coding) replaces a categorical value with a statistic
    computed from the target variable.  This is a valid technique
    **only** when the statistics are computed exclusively on the
    training fold.  Fitting the encoder on the full dataset leaks test
    labels into training features.

    This auditor detects:

    * ``TARGET_ENCODER`` operations fitted before the train/test split.
    * ``GROUPBY`` → ``AGG`` chains where the aggregation column is the
      target variable and the result is used before splitting.
    * ``LABEL_ENCODER`` fitted on the target column before splitting
      (label-encoding the target is fine, but using the encoded result
      as a feature is not).
    * Manual mean-encoding patterns via ``MERGE`` / ``JOIN`` of
      aggregated target statistics back into the feature DataFrame.

    Parameters
    ----------
    target_column_names : set[str] or None
        Explicit target column names.  Falls back to DAG metadata.
    severity_estimator : TargetLeakageSeverityEstimator or None
        Used to grade the severity of discovered patterns.
    """

    _ENCODER_OPS: frozenset[OpType] = frozenset({
        OpType.TARGET_ENCODER, OpType.LABEL_ENCODER,
        OpType.ORDINAL_ENCODER,
    })

    _GROUPBY_OPS: frozenset[OpType] = frozenset({
        OpType.GROUPBY, OpType.AGG, OpType.AGGREGATE,
    })

    _JOIN_OPS: frozenset[OpType] = frozenset({
        OpType.MERGE, OpType.JOIN,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    _FIT_OPS: frozenset[OpType] = frozenset({
        OpType.FIT, OpType.FIT_TRANSFORM, OpType.FIT_PREDICT,
    })

    def __init__(
        self,
        *,
        target_column_names: Optional[Set[str]] = None,
        severity_estimator: Optional[TargetLeakageSeverityEstimator] = None,
    ) -> None:
        self._explicit_targets: Optional[Set[str]] = (
            set(target_column_names) if target_column_names else None
        )
        self._severity = severity_estimator or TargetLeakageSeverityEstimator()

    # ----- public API ---------------------------------------------------

    def audit(self, dag: PIDAG) -> List[TargetLeakagePattern]:
        """Scan the DAG for target-encoding leakage.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[TargetLeakagePattern]
        """
        patterns: List[TargetLeakagePattern] = []
        target_cols = self._resolve_targets(dag)
        if not target_cols:
            return patterns

        split_ids = self._split_node_ids(dag)

        patterns.extend(self._check_encoder_ops(dag, target_cols, split_ids))
        patterns.extend(self._check_groupby_target(dag, target_cols, split_ids))
        patterns.extend(self._check_manual_mean_encoding(dag, target_cols, split_ids))
        patterns.extend(self._check_label_encoding_as_feature(dag, target_cols, split_ids))

        return patterns

    # ----- resolution / utilities ---------------------------------------

    def _resolve_targets(self, dag: PIDAG) -> Set[str]:
        """Determine the set of target column names."""
        if self._explicit_targets:
            return set(self._explicit_targets)
        meta_target = dag.metadata.get("target_column", None)
        if meta_target:
            if isinstance(meta_target, str):
                return {meta_target}
            if isinstance(meta_target, (list, set, frozenset)):
                return set(meta_target)
        return set()

    def _split_node_ids(self, dag: PIDAG) -> Set[str]:
        """Return the IDs of all split nodes in the DAG."""
        return {
            n.node_id for n in dag.nodes.values()
            if n.op_type in self._SPLIT_OPS
        }

    @staticmethod
    def _is_before_any_split(
        dag: PIDAG, node_id: str, split_ids: Set[str]
    ) -> bool:
        """Return *True* if *node_id* is an ancestor of any split node."""
        descendants = dag.descendants(node_id)
        return bool(descendants & split_ids)

    # ----- sub-checks ---------------------------------------------------

    def _check_encoder_ops(
        self,
        dag: PIDAG,
        target_cols: Set[str],
        split_ids: Set[str],
    ) -> List[TargetLeakagePattern]:
        """Detect TARGET_ENCODER / LABEL_ENCODER fitted before split."""
        patterns: List[TargetLeakagePattern] = []
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if node.op_type not in self._ENCODER_OPS:
                continue
            if not self._is_before_any_split(dag, node.node_id, split_ids):
                continue

            encoder_target = node.metadata.get("target_column", None)
            encoded_cols = node.metadata.get("columns", [])
            if isinstance(encoded_cols, str):
                encoded_cols = [encoded_cols]

            is_target_encoder = node.op_type == OpType.TARGET_ENCODER
            uses_target = (
                encoder_target in target_cols
                or bool(set(encoded_cols) & target_cols)
                or is_target_encoder
            )
            if not uses_target:
                continue

            affected = sorted(set(node.output_columns) - target_cols)
            if not affected:
                affected = sorted(node.output_columns)

            tc = encoder_target if encoder_target in target_cols else next(iter(target_cols))
            severity = self._severity.estimate(
                violation_type=TargetViolationType.TARGET_ENCODING_LEAKAGE,
            )

            if node.op_type == OpType.TARGET_ENCODER:
                vtype = TargetViolationType.TARGET_ENCODING_LEAKAGE
                desc = (
                    f"Target encoder at node '{node.node_id}' is fitted "
                    f"on the full dataset before the train/test split, "
                    f"leaking target statistics into training features."
                )
                fix = (
                    "Fit the target encoder only on the training fold. "
                    "Use sklearn's Pipeline or a cross-validated target "
                    "encoder to prevent leakage."
                )
            else:
                vtype = TargetViolationType.LABEL_ENCODING_ON_TARGET
                desc = (
                    f"Label/ordinal encoder at node '{node.node_id}' is "
                    f"applied to target-related columns before the "
                    f"train/test split."
                )
                fix = (
                    "Fit the encoder only on the training fold and "
                    "transform the test fold separately."
                )

            patterns.append(TargetLeakagePattern(
                description=desc,
                severity=severity,
                source_nodes=[node.node_id],
                affected_features=affected,
                violation_type=vtype,
                target_column=tc,
                feature_path=[node.node_id],
                remediation=fix,
            ))
        return patterns

    def _check_groupby_target(
        self,
        dag: PIDAG,
        target_cols: Set[str],
        split_ids: Set[str],
    ) -> List[TargetLeakagePattern]:
        """Detect groupby-aggregate operations on the target column
        that occur before splitting."""
        patterns: List[TargetLeakagePattern] = []
        if not split_ids:
            return patterns

        for node in dag.nodes.values():
            if node.op_type not in self._GROUPBY_OPS:
                continue
            if not self._is_before_any_split(dag, node.node_id, split_ids):
                continue

            agg_cols = self._extract_agg_columns(node)
            leaked = agg_cols & target_cols
            if not leaked:
                continue

            by_cols = node.metadata.get("by", node.metadata.get("keys", []))
            if isinstance(by_cols, str):
                by_cols = [by_cols]
            affected = sorted(
                set(node.output_columns) - target_cols - set(by_cols)
            )
            if not affected:
                affected = sorted(node.output_columns)

            tc = next(iter(leaked))
            patterns.append(TargetLeakagePattern(
                description=(
                    f"GroupBy aggregation at node '{node.node_id}' "
                    f"computes statistics on target column '{tc}' "
                    f"before the train/test split, creating target-"
                    f"encoded features from the full dataset."
                ),
                severity=Severity.CRITICAL,
                source_nodes=[node.node_id],
                affected_features=affected,
                violation_type=TargetViolationType.TARGET_AGGREGATION_LEAKAGE,
                target_column=tc,
                feature_path=[node.node_id],
                remediation=(
                    "Perform the groupby aggregation on the target "
                    "column only within the training fold, then merge "
                    "the result back onto training and test data "
                    "separately."
                ),
            ))
        return patterns

    def _check_manual_mean_encoding(
        self,
        dag: PIDAG,
        target_cols: Set[str],
        split_ids: Set[str],
    ) -> List[TargetLeakagePattern]:
        """Detect manual mean-encoding patterns: a GROUPBY on the target
        followed by a MERGE/JOIN that maps aggregated statistics back
        into the feature DataFrame before splitting."""
        patterns: List[TargetLeakagePattern] = []
        if not split_ids:
            return patterns

        agg_nodes_with_target: List[PipelineNode] = []
        for node in dag.nodes.values():
            if node.op_type not in self._GROUPBY_OPS:
                continue
            agg_cols = self._extract_agg_columns(node)
            if agg_cols & target_cols:
                agg_nodes_with_target.append(node)

        for agg_node in agg_nodes_with_target:
            descendants = dag.descendants(agg_node.node_id)
            for desc_id in descendants:
                desc_node = dag.get_node(desc_id)
                if desc_node.op_type not in self._JOIN_OPS:
                    continue
                if not self._is_before_any_split(dag, desc_id, split_ids):
                    continue

                affected = sorted(
                    set(desc_node.output_columns) - target_cols
                    - set(desc_node.input_columns)
                )
                if not affected:
                    affected = sorted(desc_node.output_columns)

                tc = next(iter(
                    self._extract_agg_columns(agg_node) & target_cols
                ))
                patterns.append(TargetLeakagePattern(
                    description=(
                        f"Manual mean-encoding pattern detected: "
                        f"aggregation on target '{tc}' at node "
                        f"'{agg_node.node_id}' is merged back into "
                        f"features at node '{desc_id}' before splitting."
                    ),
                    severity=Severity.CRITICAL,
                    source_nodes=[agg_node.node_id, desc_id],
                    affected_features=affected,
                    violation_type=TargetViolationType.TARGET_ENCODING_LEAKAGE,
                    target_column=tc,
                    feature_path=[agg_node.node_id, desc_id],
                    remediation=(
                        "Compute target-based statistics only on the "
                        "training fold before merging.  Consider using a "
                        "cross-validated target encoder instead of a "
                        "manual groupby-merge pattern."
                    ),
                ))
        return patterns

    def _check_label_encoding_as_feature(
        self,
        dag: PIDAG,
        target_cols: Set[str],
        split_ids: Set[str],
    ) -> List[TargetLeakagePattern]:
        """Detect label-encoded target columns that flow back into
        features.

        Label-encoding the target for model consumption is fine, but
        if the encoded result is also used as a feature input, that
        constitutes leakage.
        """
        patterns: List[TargetLeakagePattern] = []
        fit_ops = self._FIT_OPS

        for node in dag.nodes.values():
            if node.op_type != OpType.LABEL_ENCODER:
                continue
            encoded_cols = node.metadata.get("columns", [])
            if isinstance(encoded_cols, str):
                encoded_cols = [encoded_cols]
            if not (set(encoded_cols) & target_cols):
                continue

            encoded_outputs = set(node.output_columns) - set(node.input_columns)
            if not encoded_outputs:
                encoded_outputs = set(node.output_columns)

            desc_ids = dag.descendants(node.node_id)
            for desc_id in desc_ids:
                desc_node = dag.get_node(desc_id)
                if desc_node.op_type not in fit_ops:
                    continue
                feature_inputs = set(desc_node.input_columns)
                leaked = feature_inputs & encoded_outputs
                if not leaked:
                    continue

                tc = next(iter(set(encoded_cols) & target_cols))
                patterns.append(TargetLeakagePattern(
                    description=(
                        f"Label-encoded target column '{tc}' produced at "
                        f"node '{node.node_id}' is used as a feature "
                        f"input at fit node '{desc_id}'."
                    ),
                    severity=Severity.CRITICAL,
                    source_nodes=[node.node_id, desc_id],
                    affected_features=sorted(leaked),
                    violation_type=TargetViolationType.LABEL_ENCODING_ON_TARGET,
                    target_column=tc,
                    feature_path=[node.node_id, desc_id],
                    remediation=(
                        "Ensure that the label-encoded target is only "
                        "used as the label argument (y) and not as a "
                        "feature column (X)."
                    ),
                ))
        return patterns

    @staticmethod
    def _extract_agg_columns(node: PipelineNode) -> Set[str]:
        """Extract the columns being aggregated at a groupby/agg node."""
        agg_cols: Set[str] = set()

        agg_spec = node.metadata.get("agg", node.metadata.get("func", None))
        if isinstance(agg_spec, dict):
            agg_cols.update(agg_spec.keys())
        elif isinstance(agg_spec, str):
            cols_meta = node.metadata.get("columns", [])
            if isinstance(cols_meta, (list, set, frozenset)):
                agg_cols.update(cols_meta)

        target_meta = node.metadata.get("target_column", None)
        if target_meta:
            if isinstance(target_meta, str):
                agg_cols.add(target_meta)

        value_col = node.metadata.get("values", node.metadata.get("value", None))
        if isinstance(value_col, str):
            agg_cols.add(value_col)
        elif isinstance(value_col, (list, set)):
            agg_cols.update(value_col)

        return agg_cols


# ===================================================================
#  ProxyFeatureDetector
# ===================================================================


class ProxyFeatureDetector:
    """Detect features that are highly correlated with the target and
    may serve as proxies.

    A proxy feature does not *directly* contain the target, but is
    correlated strongly enough that it effectively leaks target
    information.  Examples:

    * An ``approved`` flag that is deterministic given the ``default``
      label.
    * Revenue columns that are monotonically related to the profit
      target.

    Detection works by inspecting node-level provenance annotations
    and metadata that record per-column statistical summaries.  When
    such summaries are absent, the detector falls back to heuristic
    name matching (e.g. a column named ``predicted_target`` is
    suspicious).

    Parameters
    ----------
    target_column_names : set[str] or None
        Explicit target column names.
    correlation_warn : float
        Absolute correlation threshold for ``WARNING``.  Default ``0.85``.
    correlation_critical : float
        Absolute correlation threshold for ``CRITICAL``.  Default ``0.98``.
    mi_warn : float
        MI threshold (bits) for ``WARNING``.  Default ``1.0``.
    mi_critical : float
        MI threshold (bits) for ``CRITICAL``.  Default ``5.0``.
    suspicious_name_patterns : list[str] or None
        Additional regex patterns that flag suspicious column names.
    """

    _DEFAULT_SUSPICIOUS_PATTERNS: List[str] = [
        r"(?i)predict(ed)?_",
        r"(?i)_predict(ed)?$",
        r"(?i)^prob_",
        r"(?i)_prob(ability)?$",
        r"(?i)^score_",
        r"(?i)_score$",
        r"(?i)^estimated_",
        r"(?i)_estimated$",
        r"(?i)^fitted_",
        r"(?i)_fitted$",
        r"(?i)^actual_",
        r"(?i)_actual$",
    ]

    def __init__(
        self,
        *,
        target_column_names: Optional[Set[str]] = None,
        correlation_warn: float = 0.85,
        correlation_critical: float = 0.98,
        mi_warn: float = 1.0,
        mi_critical: float = 5.0,
        suspicious_name_patterns: Optional[List[str]] = None,
    ) -> None:
        self._explicit_targets: Optional[Set[str]] = (
            set(target_column_names) if target_column_names else None
        )
        self._corr_warn = correlation_warn
        self._corr_crit = correlation_critical
        self._mi_warn = mi_warn
        self._mi_crit = mi_critical

        raw_patterns = (
            suspicious_name_patterns
            if suspicious_name_patterns is not None
            else self._DEFAULT_SUSPICIOUS_PATTERNS
        )
        self._suspicious_res: List[re.Pattern[str]] = [
            re.compile(p) for p in raw_patterns
        ]
        self._severity = TargetLeakageSeverityEstimator(
            mi_warn_threshold=mi_warn,
            mi_critical_threshold=mi_critical,
            corr_warn_threshold=correlation_warn,
            corr_critical_threshold=correlation_critical,
        )

    # ----- public API ---------------------------------------------------

    def detect(self, dag: PIDAG) -> List[TargetLeakagePattern]:
        """Scan the DAG for proxy features.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[TargetLeakagePattern]
        """
        patterns: List[TargetLeakagePattern] = []
        target_cols = self._resolve_targets(dag)
        if not target_cols:
            return patterns

        patterns.extend(self._detect_by_statistics(dag, target_cols))
        patterns.extend(self._detect_by_name_heuristics(dag, target_cols))

        seen: Set[Tuple[str, str]] = set()
        deduplicated: List[TargetLeakagePattern] = []
        for p in patterns:
            key = (p.target_column, tuple(p.affected_features).__repr__())
            if key not in seen:
                seen.add(key)
                deduplicated.append(p)
        return deduplicated

    # ----- resolution ---------------------------------------------------

    def _resolve_targets(self, dag: PIDAG) -> Set[str]:
        """Determine the set of target column names."""
        if self._explicit_targets:
            return set(self._explicit_targets)
        meta_target = dag.metadata.get("target_column", None)
        if meta_target:
            if isinstance(meta_target, str):
                return {meta_target}
            if isinstance(meta_target, (list, set, frozenset)):
                return set(meta_target)
        return set()

    # ----- statistics-based detection -----------------------------------

    def _detect_by_statistics(
        self, dag: PIDAG, target_cols: Set[str]
    ) -> List[TargetLeakagePattern]:
        """Use per-column statistical metadata to find high-correlation
        proxy features."""
        patterns: List[TargetLeakagePattern] = []

        for node in dag.nodes.values():
            col_stats = node.metadata.get("column_statistics", {})
            if not isinstance(col_stats, dict):
                continue

            for col_name, stats in col_stats.items():
                if col_name in target_cols:
                    continue
                if not isinstance(stats, dict):
                    continue

                corr_with_target = self._get_target_correlation(
                    stats, target_cols
                )
                mi_with_target = self._get_target_mi(stats, target_cols)

                if corr_with_target is None and mi_with_target is None:
                    continue

                corr_val = abs(corr_with_target) if corr_with_target is not None else 0.0
                mi_val = mi_with_target if mi_with_target is not None else 0.0

                severity = self._severity.estimate(
                    mutual_information_bits=mi_val,
                    correlation_score=corr_val,
                    violation_type=TargetViolationType.PROXY_FEATURE_CORRELATION,
                )
                if severity == Severity.NEGLIGIBLE:
                    continue

                tc = self._identify_correlated_target(stats, target_cols)
                patterns.append(TargetLeakagePattern(
                    description=(
                        f"Feature '{col_name}' at node '{node.node_id}' "
                        f"has high correlation (|r|={corr_val:.3f}) and/or "
                        f"mutual information ({mi_val:.2f} bits) with "
                        f"target '{tc}', suggesting it is a proxy feature."
                    ),
                    severity=severity,
                    source_nodes=[node.node_id],
                    affected_features=[col_name],
                    violation_type=TargetViolationType.PROXY_FEATURE_CORRELATION,
                    target_column=tc,
                    feature_path=[node.node_id],
                    correlation_score=corr_val,
                    mutual_information_bits=mi_val,
                    remediation=(
                        f"Investigate feature '{col_name}': if it is "
                        f"causally derived from the target or unavailable "
                        f"at prediction time, remove it from the feature "
                        f"set."
                    ),
                ))
        return patterns

    @staticmethod
    def _get_target_correlation(
        stats: Dict[str, Any], target_cols: Set[str]
    ) -> Optional[float]:
        """Extract correlation with any target column from stats dict."""
        corr_dict = stats.get("correlations", {})
        if not isinstance(corr_dict, dict):
            return None
        for tc in target_cols:
            if tc in corr_dict:
                try:
                    return float(corr_dict[tc])
                except (TypeError, ValueError):
                    pass
        corr_val = stats.get("target_correlation", None)
        if corr_val is not None:
            try:
                return float(corr_val)
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _get_target_mi(
        stats: Dict[str, Any], target_cols: Set[str]
    ) -> Optional[float]:
        """Extract mutual information with target from stats dict."""
        mi_dict = stats.get("mutual_information", {})
        if not isinstance(mi_dict, dict):
            return None
        for tc in target_cols:
            if tc in mi_dict:
                try:
                    return float(mi_dict[tc])
                except (TypeError, ValueError):
                    pass
        mi_val = stats.get("target_mi", stats.get("target_mutual_information", None))
        if mi_val is not None:
            try:
                return float(mi_val)
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _identify_correlated_target(
        stats: Dict[str, Any], target_cols: Set[str]
    ) -> str:
        """Return the specific target column with the highest
        correlation, or the first target column if unavailable."""
        best_tc = ""
        best_corr = -1.0
        corr_dict = stats.get("correlations", {})
        if isinstance(corr_dict, dict):
            for tc in target_cols:
                if tc in corr_dict:
                    try:
                        val = abs(float(corr_dict[tc]))
                        if val > best_corr:
                            best_corr = val
                            best_tc = tc
                    except (TypeError, ValueError):
                        pass
        if best_tc:
            return best_tc
        return next(iter(target_cols))

    # ----- name-based heuristic detection -------------------------------

    def _detect_by_name_heuristics(
        self, dag: PIDAG, target_cols: Set[str]
    ) -> List[TargetLeakagePattern]:
        """Flag columns whose names match suspicious proxy patterns."""
        patterns: List[TargetLeakagePattern] = []
        all_columns: Dict[str, List[str]] = {}

        for node in dag.nodes.values():
            for col in node.output_columns:
                if col not in target_cols:
                    all_columns.setdefault(col, []).append(node.node_id)

        for col, node_ids in all_columns.items():
            if self._name_is_suspicious(col, target_cols):
                patterns.append(TargetLeakagePattern(
                    description=(
                        f"Feature column '{col}' has a name that "
                        f"suggests it may be a proxy for the target. "
                        f"Found at node(s): {', '.join(node_ids[:3])}."
                    ),
                    severity=Severity.WARNING,
                    source_nodes=node_ids[:3],
                    affected_features=[col],
                    violation_type=TargetViolationType.PROXY_FEATURE_CORRELATION,
                    target_column=next(iter(target_cols)),
                    feature_path=node_ids[:3],
                    remediation=(
                        f"Review feature '{col}' to determine whether "
                        f"it is available at prediction time and not "
                        f"derived from the target."
                    ),
                ))
        return patterns

    def _name_is_suspicious(self, col: str, target_cols: Set[str]) -> bool:
        """Return *True* if the column name matches a proxy pattern."""
        for pattern in self._suspicious_res:
            if pattern.search(col):
                return True
        lower = col.lower()
        for tc in target_cols:
            tc_lower = tc.lower()
            if tc_lower in lower and lower != tc_lower:
                return True
        return False


# ===================================================================
#  TargetDerivedFeatureDetector
# ===================================================================


class TargetDerivedFeatureDetector:
    """Detect features that are mathematically derived from the target
    variable.

    This detector traces information flow through the DAG and
    identifies feature columns whose provenance includes the target
    column through arithmetic, aggregation, or transformation
    operations.  Examples:

    * ``residual = actual - predicted`` where ``actual`` is the target.
    * ``normalised_target = (target - mean) / std`` used as a feature.
    * ``target_ratio = target / feature_x`` joined back to features.

    Parameters
    ----------
    target_column_names : set[str] or None
        Explicit target column names.
    max_derivation_depth : int
        Maximum number of DAG hops to trace from the target column.
        Default ``10``.
    severity_estimator : TargetLeakageSeverityEstimator or None
        Used to grade the severity of discovered patterns.
    """

    _ARITHMETIC_OPS: frozenset[OpType] = frozenset({
        OpType.ASSIGN, OpType.APPLY, OpType.APPLYMAP, OpType.MAP,
        OpType.TRANSFORM,
    })

    _AGG_OPS: frozenset[OpType] = frozenset({
        OpType.GROUPBY, OpType.AGG, OpType.AGGREGATE, OpType.PIVOT,
        OpType.PIVOT_TABLE, OpType.CROSSTAB,
    })

    _COMBINE_OPS: frozenset[OpType] = frozenset({
        OpType.MERGE, OpType.JOIN, OpType.CONCAT,
    })

    _SPLIT_OPS: frozenset[OpType] = frozenset({
        OpType.TRAIN_TEST_SPLIT, OpType.KFOLD_SPLIT,
        OpType.STRATIFIED_KFOLD, OpType.GROUP_KFOLD,
        OpType.LEAVE_ONE_OUT,
    })

    _FIT_OPS: frozenset[OpType] = frozenset({
        OpType.FIT, OpType.FIT_TRANSFORM, OpType.FIT_PREDICT,
        OpType.ESTIMATOR_FIT,
    })

    def __init__(
        self,
        *,
        target_column_names: Optional[Set[str]] = None,
        max_derivation_depth: int = 10,
        severity_estimator: Optional[TargetLeakageSeverityEstimator] = None,
    ) -> None:
        self._explicit_targets: Optional[Set[str]] = (
            set(target_column_names) if target_column_names else None
        )
        self._max_depth = max(1, max_derivation_depth)
        self._severity = severity_estimator or TargetLeakageSeverityEstimator()

    # ----- public API ---------------------------------------------------

    def detect(self, dag: PIDAG) -> List[TargetLeakagePattern]:
        """Scan the DAG for target-derived features.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[TargetLeakagePattern]
        """
        patterns: List[TargetLeakagePattern] = []
        target_cols = self._resolve_targets(dag)
        if not target_cols:
            return patterns

        tainted_map = self._propagate_target_taint(dag, target_cols)
        patterns.extend(
            self._check_tainted_features(dag, target_cols, tainted_map)
        )
        patterns.extend(
            self._check_arithmetic_derivations(dag, target_cols)
        )
        return self._deduplicate(patterns)

    # ----- resolution ---------------------------------------------------

    def _resolve_targets(self, dag: PIDAG) -> Set[str]:
        """Determine the set of target column names."""
        if self._explicit_targets:
            return set(self._explicit_targets)
        meta_target = dag.metadata.get("target_column", None)
        if meta_target:
            if isinstance(meta_target, str):
                return {meta_target}
            if isinstance(meta_target, (list, set, frozenset)):
                return set(meta_target)
        return set()

    # ----- taint propagation --------------------------------------------

    def _propagate_target_taint(
        self, dag: PIDAG, target_cols: Set[str]
    ) -> Dict[str, Set[str]]:
        """Propagate taint from target columns through the DAG.

        Returns a mapping from column name to the set of node IDs where
        the tainted column appears (excluding the original target
        columns).
        """
        tainted: Dict[str, Set[str]] = {}
        topo = dag.topological_order()

        node_tainted_inputs: Dict[str, Set[str]] = {}

        for nid in topo:
            node = dag.get_node(nid)
            incoming_taint: Set[str] = set()

            for edge in dag.in_edges(nid):
                src_id = edge.source
                src_taint = node_tainted_inputs.get(src_id, set())
                incoming_taint.update(src_taint)

            input_cols = set(node.input_columns)
            target_in_inputs = input_cols & (target_cols | incoming_taint)

            if target_in_inputs and self._is_deriving_op(node):
                new_cols = set(node.output_columns) - input_cols
                derived = new_cols - target_cols
                for col in derived:
                    tainted.setdefault(col, set()).add(nid)
                node_tainted_inputs[nid] = incoming_taint | derived | (
                    input_cols & target_cols
                )
            else:
                passed_through = incoming_taint & set(node.output_columns)
                node_tainted_inputs[nid] = passed_through | (
                    set(node.output_columns) & target_cols
                )

        return tainted

    def _is_deriving_op(self, node: PipelineNode) -> bool:
        """Return *True* if the node performs an operation that could
        derive new columns from existing ones."""
        return node.op_type in (
            self._ARITHMETIC_OPS | self._AGG_OPS | self._COMBINE_OPS
        )

    def _check_tainted_features(
        self,
        dag: PIDAG,
        target_cols: Set[str],
        tainted_map: Dict[str, Set[str]],
    ) -> List[TargetLeakagePattern]:
        """Check whether tainted columns flow into fit nodes as
        features."""
        patterns: List[TargetLeakagePattern] = []
        fit_nodes = [
            n for n in dag.nodes.values()
            if n.op_type in self._FIT_OPS
        ]

        for fit_node in fit_nodes:
            feature_inputs = set(fit_node.input_columns)
            y_meta = fit_node.metadata.get("y_column", None)
            if isinstance(y_meta, str):
                feature_inputs.discard(y_meta)
            elif isinstance(y_meta, (list, set)):
                feature_inputs -= set(y_meta)

            for col in sorted(feature_inputs):
                if col in tainted_map:
                    origin_nodes = sorted(tainted_map[col])
                    tc = self._closest_target(
                        dag, origin_nodes, target_cols
                    )
                    path = origin_nodes + [fit_node.node_id]
                    patterns.append(TargetLeakagePattern(
                        description=(
                            f"Feature '{col}' at fit node "
                            f"'{fit_node.node_id}' is derived from "
                            f"target column '{tc}' via nodes "
                            f"{origin_nodes}."
                        ),
                        severity=Severity.CRITICAL,
                        source_nodes=origin_nodes + [fit_node.node_id],
                        affected_features=[col],
                        violation_type=TargetViolationType.TARGET_DERIVED_FEATURE,
                        target_column=tc,
                        feature_path=path,
                        remediation=(
                            f"Remove feature '{col}' from the feature "
                            f"matrix or redesign its computation to not "
                            f"depend on the target column '{tc}'."
                        ),
                    ))
        return patterns

    def _check_arithmetic_derivations(
        self, dag: PIDAG, target_cols: Set[str]
    ) -> List[TargetLeakagePattern]:
        """Detect explicit arithmetic expressions referencing the
        target in ASSIGN or APPLY node metadata."""
        patterns: List[TargetLeakagePattern] = []

        for node in dag.nodes.values():
            if node.op_type not in self._ARITHMETIC_OPS:
                continue

            expr = node.metadata.get("expression", "")
            func_body = node.metadata.get("func", "")
            combined = f"{expr} {func_body}"

            refs = set()
            for tc in target_cols:
                if tc in combined:
                    refs.add(tc)

            if not refs:
                continue

            new_cols = sorted(
                set(node.output_columns) - set(node.input_columns)
            )
            if not new_cols:
                new_cols = sorted(node.output_columns)

            for tc in sorted(refs):
                patterns.append(TargetLeakagePattern(
                    description=(
                        f"Node '{node.node_id}' ({node.op_type.name}) "
                        f"contains an expression referencing target "
                        f"column '{tc}', producing derived feature(s): "
                        f"{new_cols}."
                    ),
                    severity=Severity.CRITICAL,
                    source_nodes=[node.node_id],
                    affected_features=new_cols,
                    violation_type=TargetViolationType.TARGET_DERIVED_FEATURE,
                    target_column=tc,
                    feature_path=[node.node_id],
                    remediation=(
                        f"Remove the reference to target column '{tc}' "
                        f"from the expression at node '{node.node_id}'."
                    ),
                ))
        return patterns

    @staticmethod
    def _closest_target(
        dag: PIDAG, origin_nodes: List[str], target_cols: Set[str]
    ) -> str:
        """Identify which target column is closest to the origin nodes
        in the DAG, defaulting to the first target column."""
        for nid in origin_nodes:
            node = dag.get_node(nid)
            overlap = set(node.input_columns) & target_cols
            if overlap:
                return next(iter(sorted(overlap)))
        return next(iter(sorted(target_cols)))

    @staticmethod
    def _deduplicate(
        patterns: List[TargetLeakagePattern],
    ) -> List[TargetLeakagePattern]:
        """Remove duplicate patterns based on affected features and
        target column."""
        seen: Set[Tuple[str, ...]] = set()
        unique: List[TargetLeakagePattern] = []
        for p in patterns:
            key = (
                p.target_column,
                p.violation_type.name,
                tuple(sorted(p.affected_features)),
            )
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# ===================================================================
#  TargetLeakageDetector  (top-level orchestrator)
# ===================================================================


class TargetLeakageDetector:
    """Orchestrator that runs all target-leakage sub-detectors on a
    PI-DAG and returns the combined list of detected patterns.

    Usage::

        detector = TargetLeakageDetector()
        patterns = detector.detect(dag)
        for p in patterns:
            print(p.severity, p.description)

    Parameters
    ----------
    target_column_names : set[str] or None
        Explicit target column names shared with all sub-detectors.
    check_aliases : bool
        Passed to :class:`DirectTargetDetector`.  Default ``True``.
    max_derivation_depth : int
        Passed to :class:`TargetDerivedFeatureDetector`.  Default ``10``.
    correlation_warn : float
        Passed to :class:`ProxyFeatureDetector`.  Default ``0.85``.
    correlation_critical : float
        Passed to :class:`ProxyFeatureDetector`.  Default ``0.98``.
    mi_warn : float
        MI threshold (bits) for WARNING.  Default ``1.0``.
    mi_critical : float
        MI threshold (bits) for CRITICAL.  Default ``5.0``.
    severity_estimator : TargetLeakageSeverityEstimator or None
        Shared severity estimator.  One is created automatically if
        not provided.
    """

    def __init__(
        self,
        *,
        target_column_names: Optional[Set[str]] = None,
        check_aliases: bool = True,
        max_derivation_depth: int = 10,
        correlation_warn: float = 0.85,
        correlation_critical: float = 0.98,
        mi_warn: float = 1.0,
        mi_critical: float = 5.0,
        severity_estimator: Optional[TargetLeakageSeverityEstimator] = None,
    ) -> None:
        self._severity = severity_estimator or TargetLeakageSeverityEstimator(
            mi_warn_threshold=mi_warn,
            mi_critical_threshold=mi_critical,
            corr_warn_threshold=correlation_warn,
            corr_critical_threshold=correlation_critical,
        )
        self._direct = DirectTargetDetector(
            target_column_names=target_column_names,
            check_aliases=check_aliases,
        )
        self._encoding = TargetEncodingAuditor(
            target_column_names=target_column_names,
            severity_estimator=self._severity,
        )
        self._proxy = ProxyFeatureDetector(
            target_column_names=target_column_names,
            correlation_warn=correlation_warn,
            correlation_critical=correlation_critical,
            mi_warn=mi_warn,
            mi_critical=mi_critical,
        )
        self._derived = TargetDerivedFeatureDetector(
            target_column_names=target_column_names,
            max_derivation_depth=max_derivation_depth,
            severity_estimator=self._severity,
        )

    # ----- read-only accessors ------------------------------------------

    @property
    def direct_detector(self) -> DirectTargetDetector:
        """Access the underlying :class:`DirectTargetDetector`."""
        return self._direct

    @property
    def encoding_auditor(self) -> TargetEncodingAuditor:
        """Access the underlying :class:`TargetEncodingAuditor`."""
        return self._encoding

    @property
    def proxy_detector(self) -> ProxyFeatureDetector:
        """Access the underlying :class:`ProxyFeatureDetector`."""
        return self._proxy

    @property
    def derived_detector(self) -> TargetDerivedFeatureDetector:
        """Access the underlying :class:`TargetDerivedFeatureDetector`."""
        return self._derived

    @property
    def severity_estimator(self) -> TargetLeakageSeverityEstimator:
        """Access the underlying :class:`TargetLeakageSeverityEstimator`."""
        return self._severity

    # ----- public API ---------------------------------------------------

    def detect(self, dag: PIDAG) -> List[TargetLeakagePattern]:
        """Run all sub-detectors and return the combined patterns.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.

        Returns
        -------
        list[TargetLeakagePattern]
            Detected target leakage patterns, ordered by severity
            (critical first).
        """
        patterns: List[TargetLeakagePattern] = []
        patterns.extend(self._direct.detect(dag))
        patterns.extend(self._encoding.audit(dag))
        patterns.extend(self._proxy.detect(dag))
        patterns.extend(self._derived.detect(dag))

        patterns = self._deduplicate(patterns)
        patterns.sort(key=lambda p: p.severity, reverse=True)
        return patterns

    def detect_with_severity_summary(
        self, dag: PIDAG
    ) -> Tuple[List[TargetLeakagePattern], Dict[str, int]]:
        """Like :meth:`detect` but also returns a severity summary.

        Returns
        -------
        tuple[list[TargetLeakagePattern], dict[str, int]]
            ``(patterns, summary)`` where *summary* maps severity names
            to counts.
        """
        patterns = self.detect(dag)
        summary: Dict[str, int] = {
            Severity.CRITICAL.value: 0,
            Severity.WARNING.value: 0,
            Severity.NEGLIGIBLE.value: 0,
        }
        for p in patterns:
            summary[p.severity.value] = summary.get(p.severity.value, 0) + 1
        return patterns, summary

    def detect_for_columns(
        self,
        dag: PIDAG,
        columns: Set[str],
    ) -> List[TargetLeakagePattern]:
        """Run detection but only return patterns whose
        ``affected_features`` overlap with *columns*.

        Parameters
        ----------
        dag : PIDAG
            The pipeline information DAG to analyse.
        columns : set[str]
            Column names to filter on.

        Returns
        -------
        list[TargetLeakagePattern]
        """
        all_patterns = self.detect(dag)
        return [
            p for p in all_patterns
            if set(p.affected_features) & columns
        ]

    # ----- internal helpers ---------------------------------------------

    @staticmethod
    def _deduplicate(
        patterns: List[TargetLeakagePattern],
    ) -> List[TargetLeakagePattern]:
        """Remove patterns that describe the same issue.

        Two patterns are considered duplicates when they share the same
        violation type, target column, and affected-feature set.
        """
        seen: Set[Tuple[str, ...]] = set()
        unique: List[TargetLeakagePattern] = []
        for p in patterns:
            key = (
                p.violation_type.name,
                p.target_column,
                tuple(sorted(p.affected_features)),
            )
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique
