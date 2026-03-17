"""
taintflow.analysis – Worklist-based abstract interpretation engine.

This module implements the core fixpoint computation that propagates taint
through the Pipeline Information DAG (PI-DAG) using the partition-taint lattice.

Key classes:

* :class:`WorklistAnalyzer`  – the main analysis driver.
* :class:`AnalysisResult`    – per-feature leakage bounds after fixpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from taintflow.core.lattice import (
    ColumnTaintMap,
    DataFrameAbstractState,
    PartitionTaintLattice,
    TaintElement,
)
from taintflow.core.types import (
    FeatureLeakage,
    LeakageReport,
    Origin,
    Severity,
    StageLeakage,
)


@dataclass
class AnalysisResult:
    """Result of the worklist fixpoint analysis.

    Attributes:
        column_taints: Final taint state for each output column.
        iterations: Number of worklist iterations to convergence.
        converged: Whether the fixpoint was reached within the iteration limit.
        warnings: Any analysis warnings (e.g., conservative fallbacks).
    """

    column_taints: ColumnTaintMap = field(default_factory=ColumnTaintMap)
    iterations: int = 0
    converged: bool = True
    warnings: list[str] = field(default_factory=list)

    def has_leakage(self, threshold: float = 0.0) -> bool:
        """Check if any column exceeds the given bit-bound threshold."""
        for _col, elem in self.column_taints.items():
            if Origin.TEST in elem.origins and elem.bit_bound > threshold:
                return True
        return False

    def max_leakage_bits(self) -> float:
        """Return the maximum leakage bound across all columns."""
        if not self.column_taints:
            return 0.0
        return max(
            (elem.bit_bound for elem in self.column_taints.values()),
            default=0.0,
        )

    def to_report(self, severity_thresholds: Optional[Dict[str, float]] = None) -> LeakageReport:
        """Convert analysis result to a structured LeakageReport.

        Args:
            severity_thresholds: Optional mapping of severity names to bit thresholds.
                Defaults to negligible=0.01, warning=0.1, critical=1.0.

        Returns:
            A LeakageReport with per-feature leakage summaries.
        """
        if severity_thresholds is None:
            severity_thresholds = {
                "negligible": 0.01,
                "warning": 0.1,
                "critical": 1.0,
            }

        features: dict[str, FeatureLeakage] = {}
        for col_name, elem in self.column_taints.items():
            if Origin.TEST in elem.origins and elem.bit_bound > 0.0:
                if elem.bit_bound >= severity_thresholds.get("critical", 1.0):
                    severity = Severity.CRITICAL
                elif elem.bit_bound >= severity_thresholds.get("warning", 0.1):
                    severity = Severity.WARNING
                else:
                    severity = Severity.NEGLIGIBLE

                features[col_name] = FeatureLeakage(
                    column_name=col_name,
                    bit_bound=elem.bit_bound,
                    severity=severity,
                    origins=elem.origins,
                    contributing_stages=[],
                )

        # Compute aggregate severity
        max_bits = max((f.bit_bound for f in features.values()), default=0.0)
        if max_bits >= severity_thresholds.get("critical", 1.0):
            overall = Severity.CRITICAL
        elif max_bits >= severity_thresholds.get("warning", 0.1):
            overall = Severity.WARNING
        else:
            overall = Severity.NEGLIGIBLE

        return LeakageReport(
            pipeline_name="analyzed_pipeline",
            overall_severity=overall,
            total_bit_bound=sum(f.bit_bound for f in features.values()),
            n_features=len(self.column_taints),
            n_leaking_features=len(features),
        )


class WorklistAnalyzer:
    """Worklist-based fixpoint analyzer for the partition-taint lattice.

    Propagates taint through a PI-DAG using monotone transfer functions
    until convergence. Guaranteed to terminate because the lattice has
    finite height (69).

    Args:
        max_iterations: Maximum worklist iterations before forced termination.
        lattice: The partition-taint lattice instance.
    """

    def __init__(
        self,
        max_iterations: int = 10_000,
        lattice: Optional[PartitionTaintLattice] = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._lattice = lattice or PartitionTaintLattice()

    @property
    def max_iterations(self) -> int:
        """Maximum number of worklist iterations."""
        return self._max_iterations

    def analyze(
        self,
        dag_nodes: Sequence[Dict[str, Any]],
        dag_edges: Sequence[Dict[str, Any]],
        initial_state: Optional[Dict[str, TaintElement]] = None,
    ) -> AnalysisResult:
        """Run worklist fixpoint analysis on a PI-DAG.

        Args:
            dag_nodes: Sequence of node descriptors (id, kind, op_type, metadata).
            dag_edges: Sequence of edge descriptors (source, target, kind).
            initial_state: Optional initial taint for data source columns.

        Returns:
            AnalysisResult with per-column taint bounds.
        """
        state: dict[str, TaintElement] = dict(initial_state or {})
        worklist: list[str] = [n["id"] for n in dag_nodes]
        node_map = {n["id"]: n for n in dag_nodes}

        # Build adjacency: node_id → list of successor node_ids
        successors: dict[str, list[str]] = {n["id"]: [] for n in dag_nodes}
        predecessors: dict[str, list[str]] = {n["id"]: [] for n in dag_nodes}
        for edge in dag_edges:
            src, tgt = edge["source"], edge["target"]
            if src in successors:
                successors[src].append(tgt)
            if tgt in predecessors:
                predecessors[tgt].append(src)

        iterations = 0
        converged = False

        while worklist and iterations < self._max_iterations:
            node_id = worklist.pop(0)
            iterations += 1
            node = node_map.get(node_id)
            if node is None:
                continue

            # Collect incoming taint from predecessors
            incoming: list[TaintElement] = []
            for pred_id in predecessors.get(node_id, []):
                if pred_id in state:
                    incoming.append(state[pred_id])

            if not incoming:
                incoming = [self._lattice.bottom()]

            # Join all incoming taints
            joined = incoming[0]
            for t in incoming[1:]:
                joined = joined.join(t)

            # Apply transfer function (identity for now; subclasses override)
            new_taint = self._apply_transfer(node, joined)

            old_taint = state.get(node_id, self._lattice.bottom())
            if not new_taint.leq(old_taint):
                state[node_id] = old_taint.join(new_taint)
                for succ_id in successors.get(node_id, []):
                    if succ_id not in worklist:
                        worklist.append(succ_id)

        converged = len(worklist) == 0

        column_taints = ColumnTaintMap(state)
        return AnalysisResult(
            column_taints=column_taints,
            iterations=iterations,
            converged=converged,
        )

    def _apply_transfer(
        self, node: Dict[str, Any], incoming: TaintElement
    ) -> TaintElement:
        """Apply the transfer function for a DAG node.

        Default implementation is the identity (pass-through). Override
        in subclasses or plug in a channel-capacity catalog for real analysis.

        Args:
            node: Node descriptor dict.
            incoming: Joined taint from all predecessors.

        Returns:
            Output taint element.
        """
        return incoming


__all__ = [
    "AnalysisResult",
    "WorklistAnalyzer",
]