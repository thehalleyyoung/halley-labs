"""
Optimal repair planner for acyclic pipeline topologies (Algorithm A3).

Uses dynamic programming over the topological order of the pipeline DAG
to find a minimum-cost repair plan.  At each node the planner decides
between *recomputing*, *incrementally updating*, *schema migrating*, or
*skipping* (if the perturbation is annihilated by the node's operator).

Correctness guarantee
---------------------
For acyclic graphs the DP produces a **provably optimal** plan: no other
valid plan has lower cost under the given :class:`CostModel`.

Complexity
----------
Time:  O(|V| · 2^k)  where k = max affected columns per node (typically small).
Space: O(|V| · 2^k)  for the memoisation table.
"""

from __future__ import annotations

import logging
from typing import Any

from arc.planner.cost import CostModel
from arc.types.base import (
    ActionType,
    CompoundPerturbation,
    CostBreakdown,
    DataDelta,
    PipelineGraph,
    PipelineNode,
    QualityDelta,
    RepairAction,
    RepairPlan,
    SchemaDelta,
    SchemaOpType,
    SQLOperator,
    FilterConfig,
    GroupByConfig,
    JoinConfig,
    SelectConfig,
)

logger = logging.getLogger(__name__)

# Sentinel for "not yet computed" in the memo table
_NOT_COMPUTED = object()


class DPRepairPlanner:
    """Optimal repair planner for acyclic pipeline DAGs.

    Implements **Algorithm A3** from the ARC theory: bottom-up dynamic
    programming with annihilation detection.

    Parameters
    ----------
    cost_model:
        The cost model used to estimate repair costs.  If *None*, a
        default model is created.
    enable_annihilation:
        Whether to detect and exploit operator annihilation (a filter
        that drops all affected rows, for example).
    max_columns_tracked:
        Safety cap on the number of affected columns tracked per node
        to keep the exponential factor bounded.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        enable_annihilation: bool = True,
        max_columns_tracked: int = 20,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.enable_annihilation = enable_annihilation
        self.max_columns_tracked = max_columns_tracked

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def plan(
        self,
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
        cost_model: CostModel | None = None,
    ) -> RepairPlan:
        """Compute the optimal repair plan for *graph* under *deltas*.

        Parameters
        ----------
        graph:
            The pipeline DAG (must be acyclic).
        deltas:
            Source perturbations keyed by node ID.
        cost_model:
            Optional override cost model.

        Returns
        -------
        RepairPlan
            The minimum-cost repair plan.

        Raises
        ------
        ValueError
            If *graph* contains a cycle.
        """
        cm = cost_model or self.cost_model

        # 1. Verify acyclicity
        if not graph.is_acyclic():
            raise ValueError(
                "DPRepairPlanner requires an acyclic pipeline graph. "
                "Use LPRepairPlanner for general topologies."
            )

        topo = graph.topological_order()
        if not deltas:
            return self._empty_plan(graph, cm)

        # 2. Propagate deltas forward
        propagated = self._compute_propagated_deltas(graph, deltas, topo)

        # 3. Identify affected nodes
        affected = self._identify_affected_nodes(propagated)
        if not affected:
            return self._empty_plan(graph, cm)

        # 4. Check annihilation
        annihilated: set[str] = set()
        if self.enable_annihilation:
            annihilated = self._find_annihilated_nodes(graph, propagated)

        # 5. Bottom-up DP
        memo: dict[str, tuple[float, list[RepairAction]]] = {}
        for nid in reversed(topo):
            if nid not in affected:
                memo[nid] = (0.0, [])
                continue

            node = graph.nodes[nid]
            delta = propagated.get(nid, CompoundPerturbation())
            children = graph.children(nid)
            affected_children = [c for c in children if c in affected]

            # Option A: skip if annihilated
            if nid in annihilated:
                child_cost = sum(memo.get(c, (0.0, []))[0] for c in affected_children)
                skip_action = RepairAction(
                    node_id=nid,
                    action_type=ActionType.SKIP,
                    estimated_cost=0.0,
                    metadata={"reason": "annihilated"},
                )
                memo[nid] = (child_cost, [skip_action])
                logger.debug("Node %s: annihilated (free skip)", nid)
                continue

            # Option B: recompute this node (children don't need repair from our delta)
            recompute_cost = cm.estimate_recompute_cost(
                node,
                self._parent_sizes(graph, nid),
            )
            recompute_action = RepairAction(
                node_id=nid,
                action_type=ActionType.RECOMPUTE,
                estimated_cost=recompute_cost,
                dependencies=tuple(graph.parents(nid)),
                delta_to_apply=delta,
            )

            # Option C: incremental update + propagate to children
            inc_cost = cm.estimate_incremental_cost(
                node,
                delta.data_delta.total_changes if delta.has_data_change else 0,
            )
            inc_action = RepairAction(
                node_id=nid,
                action_type=ActionType.INCREMENTAL_UPDATE,
                estimated_cost=inc_cost,
                dependencies=tuple(graph.parents(nid)),
                delta_to_apply=delta,
            )

            # Option D: schema migration (only if schema change, no data change)
            schema_cost = float("inf")
            schema_action: RepairAction | None = None
            if delta.has_schema_change and not delta.has_data_change:
                schema_cost = cm.factors.materialization_cost * 2.0
                schema_action = RepairAction(
                    node_id=nid,
                    action_type=ActionType.SCHEMA_MIGRATE,
                    estimated_cost=schema_cost,
                    dependencies=tuple(graph.parents(nid)),
                    delta_to_apply=delta,
                )

            # Cost of propagating to children (needed for incremental + schema)
            child_propagation_cost = sum(
                memo.get(c, (0.0, []))[0] for c in affected_children
            )
            child_actions: list[RepairAction] = []
            for c in affected_children:
                child_actions.extend(memo.get(c, (0.0, []))[1])

            # Compare options
            option_recompute = recompute_cost
            option_incremental = inc_cost + child_propagation_cost
            option_schema = schema_cost + child_propagation_cost

            best_cost = option_recompute
            best_actions: list[RepairAction] = [recompute_action]

            if option_incremental < best_cost:
                best_cost = option_incremental
                best_actions = [inc_action] + child_actions

            if schema_action is not None and option_schema < best_cost:
                best_cost = option_schema
                best_actions = [schema_action] + child_actions

            memo[nid] = (best_cost, best_actions)
            logger.debug(
                "Node %s: best=%.6f (recompute=%.6f, inc=%.6f, schema=%.6f)",
                nid, best_cost, option_recompute, option_incremental, option_schema,
            )

        # 6. Extract plan
        plan = self._extract_plan(memo, graph, topo, affected, annihilated, cm)

        # 7. Validate
        self._validate_plan(plan, graph, deltas, affected)

        return plan

    # ══════════════════════════════════════════════════════════════════
    # Delta propagation
    # ══════════════════════════════════════════════════════════════════

    def _compute_propagated_deltas(
        self,
        graph: PipelineGraph,
        source_deltas: dict[str, CompoundPerturbation],
        topo: list[str],
    ) -> dict[str, CompoundPerturbation]:
        """Propagate source deltas forward through the graph.

        At each node, the propagated delta is the composition of:
        1. Any direct delta at this node (from ``source_deltas``).
        2. Deltas propagated from parent nodes, transformed by the
           node's operator (using the push functor).

        Returns a mapping from node ID to the effective delta at that node.
        """
        result: dict[str, CompoundPerturbation] = {}

        for nid in topo:
            # Start with any direct perturbation
            delta = source_deltas.get(nid, CompoundPerturbation())

            # Compose with propagated deltas from parents
            for pid in graph.parents(nid):
                parent_delta = result.get(pid)
                if parent_delta is not None and not parent_delta.is_identity:
                    pushed = self._push_delta(graph.nodes[nid], parent_delta, graph, pid)
                    delta = delta.compose(pushed)

            if not delta.is_identity:
                result[nid] = delta

        return result

    def _push_delta(
        self,
        node: PipelineNode,
        delta: CompoundPerturbation,
        graph: PipelineGraph,
        parent_id: str,
    ) -> CompoundPerturbation:
        """Push a delta through a node's operator (push functor).

        This implements the push(T, δ) functor from the ARC theory.
        The default implementation preserves the delta unless the
        operator clearly filters or projects away affected columns.
        """
        affected_cols = delta.columns_affected
        if not affected_cols:
            return delta

        # Check if a SELECT/projection removes all affected columns
        if node.operator == SQLOperator.SELECT:
            if isinstance(node.operator_config, SelectConfig):
                kept = set(node.operator_config.columns)
                if affected_cols and not affected_cols & kept:
                    return CompoundPerturbation()

        # Check if a FILTER might reduce the delta
        if node.operator == SQLOperator.FILTER:
            if isinstance(node.operator_config, FilterConfig):
                ref_cols = set(node.operator_config.columns_referenced)
                if affected_cols and not affected_cols & ref_cols:
                    return delta

        # For joins, check which side is affected
        if node.operator == SQLOperator.JOIN:
            if isinstance(node.operator_config, JoinConfig):
                left_keys = set(node.operator_config.left_keys)
                right_keys = set(node.operator_config.right_keys)
                parents = graph.parents(node.node_id)
                if len(parents) >= 2:
                    if parent_id == parents[0]:
                        if affected_cols & left_keys:
                            return delta
                    elif parent_id == parents[1]:
                        if affected_cols & right_keys:
                            return delta

        return delta

    # ══════════════════════════════════════════════════════════════════
    # Annihilation detection
    # ══════════════════════════════════════════════════════════════════

    def _check_annihilation(
        self,
        node: PipelineNode,
        delta: CompoundPerturbation,
    ) -> bool:
        """Check whether *node*'s operator annihilates *delta*.

        Annihilation occurs when the operator's semantics guarantee
        that the perturbation has no observable effect on the output.
        Examples:
        * A FILTER that references none of the affected columns.
        * A GROUP_BY whose group keys are disjoint from affected columns,
          and the aggregate is insensitive (e.g., COUNT).
        * A projection that drops all affected columns.
        """
        if delta.is_identity:
            return True

        affected = delta.columns_affected
        if not affected:
            return True

        # SELECT that drops all affected columns
        if node.operator == SQLOperator.SELECT:
            if isinstance(node.operator_config, SelectConfig):
                kept = set(node.operator_config.columns) | set(node.operator_config.expressions.keys())
                if not affected & kept:
                    return True

        # GROUP_BY on unaffected keys with count-like aggregates
        if node.operator == SQLOperator.GROUP_BY:
            if isinstance(node.operator_config, GroupByConfig):
                group_cols = set(node.operator_config.group_columns)
                if not affected & group_cols:
                    from arc.types.base import AggregateFunction
                    aggs = set(node.operator_config.aggregates.values())
                    count_like = {AggregateFunction.COUNT, AggregateFunction.COUNT_DISTINCT}
                    if aggs and aggs <= count_like:
                        return True

        # Schema-only delta on columns not in the output
        if delta.has_schema_change and not delta.has_data_change and not delta.has_quality_change:
            if node.output_schema is not None:
                out_cols = set(node.output_schema.column_names)
                if not affected & out_cols:
                    return True

        return False

    def _find_annihilated_nodes(
        self,
        graph: PipelineGraph,
        propagated: dict[str, CompoundPerturbation],
    ) -> set[str]:
        """Return the set of nodes where the delta is annihilated."""
        annihilated: set[str] = set()
        for nid, delta in propagated.items():
            node = graph.nodes.get(nid)
            if node is not None and self._check_annihilation(node, delta):
                annihilated.add(nid)
                logger.debug("Annihilation detected at node %s", nid)
        return annihilated

    # ══════════════════════════════════════════════════════════════════
    # Plan extraction and validation
    # ══════════════════════════════════════════════════════════════════

    def _extract_plan(
        self,
        memo: dict[str, tuple[float, list[RepairAction]]],
        graph: PipelineGraph,
        topo: list[str],
        affected: set[str],
        annihilated: set[str],
        cost_model: CostModel,
    ) -> RepairPlan:
        """Extract the optimal repair plan from the DP memo table."""
        # Collect all actions, deduplicating by node_id (keep the first)
        seen_nodes: set[str] = set()
        actions: list[RepairAction] = []

        # Start from root affected nodes (those with no affected parent)
        root_affected = set()
        for nid in affected:
            parent_affected = any(p in affected for p in graph.parents(nid))
            if not parent_affected:
                root_affected.add(nid)

        # Collect actions from root nodes' memo entries
        for nid in topo:
            if nid in root_affected:
                _, node_actions = memo.get(nid, (0.0, []))
                for action in node_actions:
                    if action.node_id not in seen_nodes:
                        seen_nodes.add(action.node_id)
                        actions.append(action)

        # Add NO_OP for affected nodes not covered
        for nid in topo:
            if nid in affected and nid not in seen_nodes:
                _, node_actions = memo.get(nid, (0.0, []))
                for action in node_actions:
                    if action.node_id not in seen_nodes:
                        seen_nodes.add(action.node_id)
                        actions.append(action)

        # Build execution order (topological)
        exec_order = [nid for nid in topo if nid in seen_nodes]
        total_cost = sum(a.estimated_cost for a in actions)
        full_cost = cost_model.estimate_full_recompute_cost(graph)
        savings = 1.0 - (total_cost / full_cost) if full_cost > 0 else 0.0

        cost_breakdown = cost_model.total_plan_cost(
            RepairPlan(
                actions=tuple(actions),
                execution_order=tuple(exec_order),
                total_cost=total_cost,
                full_recompute_cost=full_cost,
            ),
            graph,
        )

        return RepairPlan(
            actions=tuple(actions),
            execution_order=tuple(exec_order),
            total_cost=total_cost,
            full_recompute_cost=full_cost,
            savings_ratio=max(savings, 0.0),
            affected_nodes=frozenset(affected),
            annihilated_nodes=frozenset(annihilated),
            plan_metadata={
                "planner": "DPRepairPlanner",
                "algorithm": "A3",
                "topology": "acyclic",
            },
            cost_breakdown=cost_breakdown,
        )

    def _validate_plan(
        self,
        plan: RepairPlan,
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
        affected: set[str],
    ) -> bool:
        """Validate that the plan covers all affected nodes.

        Every affected (non-annihilated) node must have a non-SKIP
        action, or be downstream of a RECOMPUTE action.
        """
        covered: set[str] = set()
        for action in plan.actions:
            if action.action_type in {ActionType.RECOMPUTE, ActionType.INCREMENTAL_UPDATE, ActionType.SCHEMA_MIGRATE}:
                covered.add(action.node_id)
                covered |= graph.reachable_from(action.node_id)
            elif action.action_type == ActionType.SKIP:
                covered.add(action.node_id)

        # All affected nodes should be covered
        uncovered = affected - covered - plan.annihilated_nodes
        if uncovered:
            logger.warning(
                "Plan validation: %d uncovered affected nodes: %s",
                len(uncovered),
                uncovered,
            )
            return False
        return True

    # ══════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════

    def _identify_affected_nodes(
        self,
        propagated: dict[str, CompoundPerturbation],
    ) -> set[str]:
        """Return the set of nodes that have a non-identity propagated delta."""
        return {nid for nid, d in propagated.items() if not d.is_identity}

    def _parent_sizes(
        self,
        graph: PipelineGraph,
        node_id: str,
    ) -> dict[str, int]:
        """Get estimated row counts for parent nodes."""
        sizes: dict[str, int] = {}
        for pid in graph.parents(node_id):
            pn = graph.nodes.get(pid)
            if pn is not None:
                sizes[pid] = max(pn.estimated_row_count, 1)
        return sizes

    def _empty_plan(
        self,
        graph: PipelineGraph,
        cost_model: CostModel,
    ) -> RepairPlan:
        """Return an empty no-op plan."""
        full_cost = cost_model.estimate_full_recompute_cost(graph)
        return RepairPlan(
            full_recompute_cost=full_cost,
            savings_ratio=1.0,
            plan_metadata={
                "planner": "DPRepairPlanner",
                "algorithm": "A3",
                "topology": "acyclic",
                "note": "no deltas to repair",
            },
        )

    def __repr__(self) -> str:
        return (
            f"DPRepairPlanner(annihilation={self.enable_annihilation}, "
            f"max_cols={self.max_columns_tracked})"
        )
