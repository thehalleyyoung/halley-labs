"""
Cost model for repair operations in the Algebraic Repair Calculus.

Provides :class:`CostModel` which estimates the computational cost
of recomputing, incrementally updating, or migrating individual pipeline
nodes.  Cost factors are tuneable and account for CPU, I/O,
materialisation, and network transfer.

The operator-complexity model mirrors standard database cost estimation:

* ``SELECT`` / ``FILTER``: O(n)
* ``JOIN`` (hash): O(n+m) ;  (nested-loop): O(n·m)
* ``GROUP_BY`` (hash): O(n) ;  (sort): O(n log n)
* ``WINDOW``: O(n log n) per partition
* ``UNION``: O(n+m)
"""

from __future__ import annotations

import math
from typing import Any

import attr

from arc.types.base import (
    CostBreakdown,
    CompoundPerturbation,
    PipelineGraph,
    PipelineNode,
    RepairAction,
    RepairPlan,
    Schema,
    SQLOperator,
    ActionType,
    JoinConfig,
    GroupByConfig,
    WindowConfig,
)


# ─────────────────────────────────────────────────────────────────────
# Cost factors (tuneable knobs)
# ─────────────────────────────────────────────────────────────────────

@attr.s(frozen=True, slots=True, auto_attribs=True)
class CostFactors:
    """Tuneable cost-model parameters.

    Each factor represents the cost per unit of a specific resource
    dimension.  The default values are calibrated for a single-machine
    DuckDB workload.

    Attributes
    ----------
    compute_cost_per_row:
        CPU cost per row processed (unitless, default 1e-6).
    io_cost_per_byte:
        I/O cost per byte read or written (default 1e-9).
    materialization_cost:
        Fixed cost of writing an intermediate result to disk (default 0.01).
    network_cost_per_byte:
        Network transfer cost per byte (default 5e-9).
    hash_join_factor:
        Multiplier applied when hash-join is used instead of nested-loop
        (default 1.2 — accounts for hash-table overhead).
    sort_factor:
        Multiplier for sort-based operators such as ORDER BY and
        sort-GROUP-BY (default 1.5).
    parallelism_speedup:
        Maximum theoretical speedup from parallelism (Amdahl-style
        capping, default 4.0).
    recompute_overhead:
        Fixed overhead ratio added to any full recomputation
        (default 1.1 — 10 % overhead).
    incremental_discount:
        Fraction of the full-recompute cost expected for an incremental
        update (default 0.3 — 30 % of full cost).
    """

    compute_cost_per_row: float = 1e-6
    io_cost_per_byte: float = 1e-9
    materialization_cost: float = 0.01
    network_cost_per_byte: float = 5e-9
    hash_join_factor: float = 1.2
    sort_factor: float = 1.5
    parallelism_speedup: float = 4.0
    recompute_overhead: float = 1.1
    incremental_discount: float = 0.3

    # -- Factory helpers ------------------------------------------------

    @classmethod
    def default(cls) -> "CostFactors":
        """Return the default cost factors."""
        return cls()

    @classmethod
    def fast_local(cls) -> "CostFactors":
        """Factors tuned for fast local SSD workloads."""
        return cls(
            compute_cost_per_row=5e-7,
            io_cost_per_byte=5e-10,
            materialization_cost=0.005,
            network_cost_per_byte=0.0,
        )

    @classmethod
    def cloud(cls) -> "CostFactors":
        """Factors tuned for cloud/network-attached storage."""
        return cls(
            compute_cost_per_row=2e-6,
            io_cost_per_byte=5e-9,
            materialization_cost=0.05,
            network_cost_per_byte=1e-8,
        )

    def scale_compute(self, factor: float) -> "CostFactors":
        """Return a copy with compute cost scaled by *factor*."""
        return attr.evolve(self, compute_cost_per_row=self.compute_cost_per_row * factor)

    def scale_io(self, factor: float) -> "CostFactors":
        """Return a copy with I/O cost scaled by *factor*."""
        return attr.evolve(self, io_cost_per_byte=self.io_cost_per_byte * factor)


# ─────────────────────────────────────────────────────────────────────
# Main cost model
# ─────────────────────────────────────────────────────────────────────

class CostModel:
    """Cost estimation for repair operations.

    The model combines operator-complexity analysis with tuneable
    :class:`CostFactors` to produce scalar cost estimates for
    individual nodes and complete repair plans.

    Parameters
    ----------
    factors:
        Cost factors to use.  Defaults to :meth:`CostFactors.default`.
    """

    def __init__(self, factors: CostFactors | None = None) -> None:
        self.factors = factors or CostFactors.default()
        self._complexity_cache: dict[str, float] = {}

    # ── Public estimation methods ──────────────────────────────────────

    def estimate_recompute_cost(
        self,
        node: PipelineNode,
        input_sizes: dict[str, int] | None = None,
    ) -> float:
        """Estimate the cost of fully recomputing *node*.

        Parameters
        ----------
        node:
            The pipeline node to recompute.
        input_sizes:
            Mapping from parent node IDs to their row counts.
            Falls back to ``node.estimated_row_count`` when absent.

        Returns
        -------
        float
            Scalar cost estimate.
        """
        sizes = list((input_sizes or {}).values())
        if not sizes:
            sizes = [max(node.estimated_row_count, 1)]
        complexity = self.operator_complexity(node.operator, sizes, node)
        compute = complexity * self.factors.compute_cost_per_row
        io = self.estimate_io_cost(node, self._estimate_data_size(node, sizes))
        mat = self.estimate_materialization_cost_for_node(node, sizes)
        total = (compute + io + mat) * self.factors.recompute_overhead
        return max(total, 1e-12)

    def estimate_incremental_cost(
        self,
        node: PipelineNode,
        delta_size: int,
    ) -> float:
        """Estimate the cost of an incremental update at *node*.

        The incremental cost is modelled as a fraction of the full
        recompute cost, scaled by the delta size relative to the total
        input.

        Parameters
        ----------
        node:
            Target pipeline node.
        delta_size:
            Number of rows in the delta.

        Returns
        -------
        float
            Scalar cost estimate.
        """
        if delta_size <= 0:
            return 0.0
        base = max(node.estimated_row_count, 1)
        ratio = min(delta_size / base, 1.0)
        full_cost = self.estimate_recompute_cost(node)
        incremental = full_cost * ratio * self.factors.incremental_discount
        # Minimum: pay for the delta rows' compute
        min_cost = delta_size * self.factors.compute_cost_per_row
        return max(incremental, min_cost, 1e-12)

    def estimate_io_cost(
        self,
        node: PipelineNode,
        data_size: int,
    ) -> float:
        """Estimate the I/O cost for reading/writing *data_size* bytes at *node*.

        Accounts for both read and write I/O (factor of 2).
        """
        return data_size * self.factors.io_cost_per_byte * 2.0

    def estimate_materialization_cost(
        self,
        schema: Schema,
        row_count: int,
    ) -> float:
        """Estimate the cost of materialising *row_count* rows with *schema*.

        Combines the fixed materialisation overhead with per-byte I/O.
        """
        if row_count <= 0:
            return 0.0
        row_width = self._schema_row_width(schema)
        data_bytes = row_width * row_count
        return (
            self.factors.materialization_cost
            + data_bytes * self.factors.io_cost_per_byte
        )

    def estimate_materialization_cost_for_node(
        self,
        node: PipelineNode,
        input_sizes: list[int],
    ) -> float:
        """Materialisation cost for *node*, derived from its output schema."""
        if node.output_schema is not None:
            estimated_rows = self._estimate_output_rows(node, input_sizes)
            return self.estimate_materialization_cost(node.output_schema, estimated_rows)
        return self.factors.materialization_cost

    # ── Operator complexity ────────────────────────────────────────────

    def operator_complexity(
        self,
        op: SQLOperator,
        input_sizes: list[int],
        node: PipelineNode | None = None,
    ) -> float:
        """Return the *row-operation count* for the given operator.

        This is the dominant term in the cost model.

        Complexity rules
        ~~~~~~~~~~~~~~~~
        * ``SELECT``, ``FILTER``, ``DISTINCT``, ``LIMIT``: O(n)
        * ``JOIN`` (hash): O(n+m), (nested-loop): O(n·m)
        * ``GROUP_BY`` (hash): O(n), (sort): O(n log n)
        * ``ORDER_BY``: O(n log n)
        * ``WINDOW``: O(n log n) per partition
        * ``UNION``: O(n+m)
        * ``INSERT``, ``UPDATE``, ``DELETE``: O(n)
        * DDL / ``CUSTOM``: O(1)
        """
        n = input_sizes[0] if input_sizes else 1
        m = input_sizes[1] if len(input_sizes) > 1 else n

        if op in {
            SQLOperator.SELECT,
            SQLOperator.FILTER,
            SQLOperator.DISTINCT,
            SQLOperator.LIMIT,
            SQLOperator.INSERT,
            SQLOperator.UPDATE,
            SQLOperator.DELETE,
        }:
            return float(n)

        if op == SQLOperator.JOIN:
            use_hash = True
            if node is not None and isinstance(node.operator_config, JoinConfig):
                use_hash = node.operator_config.use_hash
            if use_hash:
                return float(n + m) * self.factors.hash_join_factor
            return float(n) * float(m)

        if op == SQLOperator.GROUP_BY:
            use_hash = True
            if node is not None and isinstance(node.operator_config, GroupByConfig):
                use_hash = not node.operator_config.having_predicate
            if use_hash:
                return float(n)
            return float(n) * math.log2(max(n, 2)) * self.factors.sort_factor

        if op == SQLOperator.ORDER_BY:
            return float(n) * math.log2(max(n, 2)) * self.factors.sort_factor

        if op == SQLOperator.WINDOW:
            partitions = 1
            if node is not None and isinstance(node.operator_config, WindowConfig):
                pc = node.operator_config.partition_columns
                partitions = max(len(pc), 1)
            per_partition = float(n) / partitions
            return partitions * per_partition * math.log2(max(per_partition, 2)) * self.factors.sort_factor

        if op in {SQLOperator.UNION, SQLOperator.INTERSECT, SQLOperator.EXCEPT}:
            return float(n + m)

        if op in {
            SQLOperator.CREATE_TABLE,
            SQLOperator.ALTER_TABLE,
            SQLOperator.DROP_TABLE,
        }:
            return 1.0

        return float(n)

    # ── Plan-level cost ────────────────────────────────────────────────

    def total_plan_cost(self, plan: RepairPlan, graph: PipelineGraph | None = None) -> CostBreakdown:
        """Compute a :class:`CostBreakdown` for the entire repair plan.

        Parameters
        ----------
        plan:
            The repair plan to cost.
        graph:
            Optional pipeline graph (used for node lookups).

        Returns
        -------
        CostBreakdown
        """
        total_compute = 0.0
        total_io = 0.0
        total_mat = 0.0
        total_net = 0.0
        per_node: dict[str, float] = {}

        for action in plan.actions:
            if action.is_noop:
                per_node[action.node_id] = 0.0
                continue
            node = self._lookup_node(action.node_id, graph)
            cost = self._cost_action(action, node, graph)
            per_node[action.node_id] = cost
            total_compute += cost * 0.7
            total_io += cost * 0.2
            total_mat += cost * 0.1

        total = total_compute + total_io + total_mat + total_net
        full_recompute = plan.full_recompute_cost if plan.full_recompute_cost > 0 else total * 1.5
        savings = max(0.0, full_recompute - total)

        return CostBreakdown(
            compute_cost=total_compute,
            io_cost=total_io,
            materialization_cost=total_mat,
            network_cost=total_net,
            total_cost=total,
            cost_per_node=per_node,
            savings_vs_full_recompute=savings,
        )

    def estimate_full_recompute_cost(self, graph: PipelineGraph) -> float:
        """Estimate the cost of fully recomputing all nodes in *graph*."""
        total = 0.0
        for nid in graph.topological_order():
            node = graph.nodes[nid]
            parent_sizes: dict[str, int] = {}
            for pid in graph.parents(nid):
                pnode = graph.nodes.get(pid)
                if pnode is not None:
                    parent_sizes[pid] = max(pnode.estimated_row_count, 1)
            total += self.estimate_recompute_cost(node, parent_sizes)
        return total

    def choose_action_type(
        self,
        node: PipelineNode,
        delta: CompoundPerturbation,
        input_sizes: dict[str, int] | None = None,
    ) -> ActionType:
        """Heuristically choose the cheapest action type for *node*.

        Returns
        -------
        ActionType
            The recommended repair action type.
        """
        if delta.is_identity:
            return ActionType.NO_OP

        if delta.has_schema_change and not delta.has_data_change:
            return ActionType.SCHEMA_MIGRATE

        recompute = self.estimate_recompute_cost(node, input_sizes)
        delta_size = delta.data_delta.total_changes if delta.has_data_change else 0
        incremental = self.estimate_incremental_cost(node, delta_size)

        if incremental < recompute * 0.8:
            return ActionType.INCREMENTAL_UPDATE
        return ActionType.RECOMPUTE

    # ── Private helpers ────────────────────────────────────────────────

    def _cost_action(
        self,
        action: RepairAction,
        node: PipelineNode | None,
        graph: PipelineGraph | None,
    ) -> float:
        """Estimate cost for a single action."""
        if action.estimated_cost > 0:
            return action.estimated_cost
        if node is None:
            return self.factors.materialization_cost

        input_sizes: dict[str, int] = {}
        if graph is not None:
            for pid in graph.parents(action.node_id):
                pn = graph.nodes.get(pid)
                if pn is not None:
                    input_sizes[pid] = max(pn.estimated_row_count, 1)

        if action.action_type == ActionType.RECOMPUTE:
            return self.estimate_recompute_cost(node, input_sizes)
        elif action.action_type == ActionType.INCREMENTAL_UPDATE:
            ds = 0
            if action.delta_to_apply is not None:
                ds = action.delta_to_apply.data_delta.total_changes
            return self.estimate_incremental_cost(node, ds)
        elif action.action_type == ActionType.SCHEMA_MIGRATE:
            return self.factors.materialization_cost * 2.0
        else:
            return self.factors.materialization_cost

    def _estimate_data_size(
        self,
        node: PipelineNode,
        input_sizes: list[int],
    ) -> int:
        """Rough estimate of data bytes processed."""
        rows = max(sum(input_sizes), 1) if input_sizes else max(node.estimated_row_count, 1)
        row_width = self._node_row_width(node)
        return rows * row_width

    def _estimate_output_rows(
        self,
        node: PipelineNode,
        input_sizes: list[int],
    ) -> int:
        """Rough estimate of output rows."""
        n = input_sizes[0] if input_sizes else max(node.estimated_row_count, 1)
        m = input_sizes[1] if len(input_sizes) > 1 else 0

        if node.operator == SQLOperator.FILTER:
            return max(n // 3, 1)
        if node.operator == SQLOperator.JOIN:
            return max(n, m, 1)
        if node.operator == SQLOperator.GROUP_BY:
            return max(n // 10, 1)
        if node.operator in {SQLOperator.UNION, SQLOperator.INTERSECT, SQLOperator.EXCEPT}:
            return n + m
        if node.operator == SQLOperator.LIMIT:
            return min(n, 1000)
        return n

    @staticmethod
    def _schema_row_width(schema: Schema) -> int:
        """Approximate row byte width from schema."""
        total = 0
        for col in schema.columns:
            base = col.sql_type.base_type if hasattr(col.sql_type, "base_type") else None
            if base is not None and hasattr(base, "value"):
                name = base.value.upper()
            else:
                name = ""
            if "INT" in name:
                total += 8
            elif "FLOAT" in name or "DOUBLE" in name or "REAL" in name or "DECIMAL" in name or "NUMERIC" in name:
                total += 8
            elif "BOOL" in name:
                total += 1
            elif "DATE" in name or "TIME" in name:
                total += 8
            elif "TEXT" in name:
                total += 256
            elif "VARCHAR" in name or "CHAR" in name:
                total += 64
            elif "JSON" in name:
                total += 256
            else:
                total += 32
        return max(total, 8)

    @staticmethod
    def _node_row_width(node: PipelineNode) -> int:
        """Row byte width from a node's output schema if available."""
        if node.output_schema is not None:
            return CostModel._schema_row_width(node.output_schema)
        if node.input_schema is not None:
            return CostModel._schema_row_width(node.input_schema)
        return 64

    @staticmethod
    def _lookup_node(
        node_id: str,
        graph: PipelineGraph | None,
    ) -> PipelineNode | None:
        """Look up a node in the graph, or return None."""
        if graph is None:
            return None
        return graph.nodes.get(node_id)

    # ── Repr ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"CostModel(factors={self.factors!r})"
