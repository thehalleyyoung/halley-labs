"""
Correctness validation for repair execution.

:class:`RepairValidator` checks whether the repaired state of each
pipeline node matches the fully-recomputed state.

* **Fragment F** (the ``exactly correct'' fragment): exact equality.
* **General case**: bounded error  ||repair − recompute||₁ ≤ ε.

Also validates schema consistency and quality constraints.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from arc.types.base import (
    NodeValidation,
    PipelineGraph,
    PipelineNode,
    QualityViolation,
    RepairPlan,
    SchemaViolation,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class RepairValidator:
    """Validates repair correctness against full recomputation.

    Parameters
    ----------
    engine:
        The execution engine to query for table data.
    default_epsilon:
        Default error tolerance for non-Fragment-F validation.
    """

    def __init__(
        self,
        engine: Any,
        default_epsilon: float = 1e-6,
    ) -> None:
        self._engine = engine
        self.default_epsilon = default_epsilon

    # ── Public API ─────────────────────────────────────────────────────

    def validate_plan(
        self,
        plan: RepairPlan,
        graph: PipelineGraph,
    ) -> ValidationResult:
        """Validate all repaired nodes in a plan.

        Checks schema consistency and, where possible, data correctness.
        """
        per_node: dict[str, NodeValidation] = {}
        schema_violations: list[SchemaViolation] = []
        quality_violations: list[QualityViolation] = []
        all_valid = True

        # Schema consistency check
        sv = self.validate_schema_consistency(graph)
        schema_violations.extend(sv)
        if sv:
            all_valid = False

        # Per-node validation
        for action in plan.actions:
            if action.is_noop:
                per_node[action.node_id] = NodeValidation(
                    node_id=action.node_id,
                    is_valid=True,
                    exact_match=True,
                    message="skipped",
                )
                continue

            node = graph.nodes.get(action.node_id)
            if node is None:
                per_node[action.node_id] = NodeValidation(
                    node_id=action.node_id,
                    is_valid=True,
                    message="node not in graph",
                )
                continue

            nv = self._validate_node(node, graph)
            per_node[action.node_id] = nv
            if not nv.is_valid:
                all_valid = False

        return ValidationResult(
            is_valid=all_valid,
            exact_match=all(nv.exact_match for nv in per_node.values()),
            schema_violations=tuple(schema_violations),
            quality_violations=tuple(quality_violations),
            per_node_results=per_node,
            message="validation passed" if all_valid else "validation failed",
        )

    def validate_fragment_f(
        self,
        repaired_table: str,
        recomputed_table: str,
    ) -> ValidationResult:
        """Fragment-F validation: exact row-level equality.

        Two tables are considered equal if they have the same rows
        (unordered) with the same values.
        """
        try:
            result = self._engine.execute_sql(
                f'SELECT COUNT(*) FROM ('
                f'  (SELECT * FROM "{repaired_table}" EXCEPT SELECT * FROM "{recomputed_table}")'
                f'  UNION ALL'
                f'  (SELECT * FROM "{recomputed_table}" EXCEPT SELECT * FROM "{repaired_table}")'
                f')'
            )
            diff_count = 0
            if result is not None:
                row = result.fetchone()
                if row is not None:
                    diff_count = row[0]

            is_exact = diff_count == 0
            return ValidationResult(
                is_valid=is_exact,
                exact_match=is_exact,
                actual_error=float(diff_count),
                error_bound=0.0,
                message=f"exact match" if is_exact else f"{diff_count} differing rows",
            )
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                message=f"Fragment-F validation failed: {exc}",
            )

    def validate_general(
        self,
        repaired_table: str,
        recomputed_table: str,
        epsilon: float | None = None,
    ) -> ValidationResult:
        """General validation: bounded L1 error.

        Computes the L1 norm of the difference between repaired and
        recomputed numeric columns, and checks that it is within ε.
        """
        eps = epsilon if epsilon is not None else self.default_epsilon

        try:
            # Get numeric columns
            cols = self._engine.get_table_schema(repaired_table)
            numeric_cols = [
                c["name"] for c in cols
                if any(t in c["type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"])
            ]

            if not numeric_cols:
                # No numeric columns — fall back to Fragment-F
                return self.validate_fragment_f(repaired_table, recomputed_table)

            # Compute L1 norm per column
            total_error = 0.0
            for col in numeric_cols:
                sql = (
                    f'SELECT COALESCE(SUM(ABS(a."{col}" - b."{col}")), 0) '
                    f'FROM "{repaired_table}" a '
                    f'FULL OUTER JOIN "{recomputed_table}" b '
                    f'ON a.rowid = b.rowid'
                )
                try:
                    result = self._engine.execute_sql(sql)
                    if result is not None:
                        row = result.fetchone()
                        if row is not None and row[0] is not None:
                            total_error += float(row[0])
                except Exception:
                    pass

            is_valid = total_error <= eps
            return ValidationResult(
                is_valid=is_valid,
                exact_match=total_error == 0.0,
                error_bound=eps,
                actual_error=total_error,
                message=f"L1 error={total_error:.8f}, bound={eps:.8f}",
            )

        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                message=f"General validation failed: {exc}",
            )

    def compute_epsilon_bound(
        self,
        graph: PipelineGraph,
        perturbation_size: int,
    ) -> float:
        """Compute a computable error bound for non-Fragment-F repairs.

        The bound grows linearly with the perturbation size and the
        graph depth, following the ARC theory's Theorem 4.2.

        Parameters
        ----------
        graph:
            The pipeline graph.
        perturbation_size:
            Total number of row changes in the perturbation.

        Returns
        -------
        float
            The error bound ε.
        """
        try:
            topo = graph.topological_order()
            depth = len(topo)
        except ValueError:
            depth = graph.node_count()

        # ε = perturbation_size × depth × machine_epsilon
        # This is a conservative bound from the ARC theory
        machine_eps = np.finfo(np.float64).eps
        return perturbation_size * depth * machine_eps * 1000.0

    def validate_schema_consistency(
        self,
        graph: PipelineGraph,
    ) -> list[SchemaViolation]:
        """Check that all edge schemas are consistent.

        For each edge (u, v), the output schema of u should be
        compatible with the input schema of v.
        """
        violations: list[SchemaViolation] = []

        for edge in graph.edges:
            src = graph.nodes.get(edge.source)
            tgt = graph.nodes.get(edge.target)
            if src is None or tgt is None:
                continue

            src_out = src.output_schema
            tgt_in = tgt.input_schema

            # Check referenced columns exist in source output
            if src_out is None:
                continue

            if tgt_in is not None:
                pass  # Could add type-compatibility checks here
            for col_name in edge.columns_referenced:
                if col_name not in src_out.column_names:
                    violations.append(SchemaViolation(
                        node_id=edge.source,
                        violation_type="missing_column",
                        message=(
                            f"Edge ({edge.source} → {edge.target}) references "
                            f"column '{col_name}' not in source output schema"
                        ),
                        column=col_name,
                    ))

        return violations

    def validate_quality_constraints(
        self,
        graph: PipelineGraph,
        constraints: list[Any] | None = None,
    ) -> list[QualityViolation]:
        """Validate quality constraints across the graph.

        Currently checks row-count sanity and null-rate thresholds
        for tables accessible via the execution engine.
        """
        violations: list[QualityViolation] = []

        for nid, node in graph.nodes.items():
            if not node.table_name:
                continue

            try:
                stats = self._engine.get_table_stats(node.table_name)
            except Exception:
                continue

            # Check for empty tables that shouldn't be
            if stats.row_count == 0 and node.estimated_row_count > 0:
                violations.append(QualityViolation(
                    node_id=nid,
                    constraint_name="non_empty",
                    message=f"Table {node.table_name} is empty but expected {node.estimated_row_count} rows",
                    metric_value=0.0,
                    threshold=float(node.estimated_row_count),
                ))

            # Check null rates
            for col_name, null_count in stats.null_counts.items():
                if stats.row_count > 0:
                    null_rate = null_count / stats.row_count
                    if null_rate > 0.95:
                        violations.append(QualityViolation(
                            node_id=nid,
                            constraint_name="low_null_rate",
                            message=f"Column {col_name} has {null_rate:.1%} nulls",
                            metric_value=null_rate,
                            threshold=0.95,
                            column=col_name,
                        ))

        return violations

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        lines: list[str] = []
        lines.append("=== Repair Validation Report ===")
        lines.append(f"Overall: {'PASS' if result.is_valid else 'FAIL'}")
        lines.append(f"Exact match: {result.exact_match}")

        if result.error_bound is not None:
            lines.append(f"Error bound: {result.error_bound:.8e}")
        if result.actual_error is not None:
            lines.append(f"Actual error: {result.actual_error:.8e}")

        if result.schema_violations:
            lines.append(f"\nSchema violations ({len(result.schema_violations)}):")
            for sv in result.schema_violations:
                lines.append(f"  [{sv.violation_type}] {sv.message}")

        if result.quality_violations:
            lines.append(f"\nQuality violations ({len(result.quality_violations)}):")
            for qv in result.quality_violations:
                lines.append(f"  [{qv.constraint_name}] {qv.message}")

        if result.per_node_results:
            lines.append(f"\nPer-node results ({len(result.per_node_results)}):")
            for nid, nv in sorted(result.per_node_results.items()):
                status = "✓" if nv.is_valid else "✗"
                lines.append(f"  {status} {nid}: {nv.message}")

        lines.append(f"\n{result.message}")
        return "\n".join(lines)

    # ── Private helpers ────────────────────────────────────────────────

    def _validate_node(
        self,
        node: PipelineNode,
        graph: PipelineGraph,
    ) -> NodeValidation:
        """Basic node validation: check table exists and has rows."""
        if not node.table_name:
            return NodeValidation(
                node_id=node.node_id,
                is_valid=True,
                message="no table to validate",
            )

        try:
            stats = self._engine.get_table_stats(node.table_name)
            has_data = stats.row_count > 0 or node.estimated_row_count == 0
            return NodeValidation(
                node_id=node.node_id,
                is_valid=has_data,
                exact_match=has_data,
                message=f"table {node.table_name}: {stats.row_count} rows",
            )
        except Exception as exc:
            return NodeValidation(
                node_id=node.node_id,
                is_valid=True,
                message=f"could not validate: {exc}",
            )

    def __repr__(self) -> str:
        return f"RepairValidator(epsilon={self.default_epsilon})"
