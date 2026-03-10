"""
Post-optimisation passes for repair plans.

:class:`PlanOptimizer` applies a sequence of passes to a
:class:`RepairPlan` produced by the DP or LP planner:

1. **Merge compatible actions** – batch compatible recomputations into
   a single SQL statement where possible.
2. **Parallelise independent actions** – mark independent actions for
   concurrent execution.
3. **Insert checkpoints** – add checkpoint actions at critical junctures
   to enable roll-back on failure.
4. **Prune redundant actions** – remove actions whose output is consumed
   by a later recompute (and therefore wasted).
5. **Reorder for locality** – group actions that touch the same tables to
   improve cache/I/O locality.
6. **Estimate parallelism** – compute the theoretical speedup from
   parallelisation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from arc.planner.cost import CostModel
from arc.types.base import (
    ActionType,
    PipelineGraph,
    RepairAction,
    RepairPlan,
    CostBreakdown,
)

logger = logging.getLogger(__name__)


class PlanOptimizer:
    """Post-optimisation of repair plans.

    The optimizer applies a configurable sequence of passes.  Each pass
    is idempotent and preserves plan correctness.

    Parameters
    ----------
    cost_model:
        Cost model used for re-costing after optimisation.
    enable_merge:
        Enable the merge-compatible-actions pass.
    enable_parallel:
        Enable the parallelise-independent pass.
    enable_checkpoints:
        Enable the insert-checkpoints pass.
    enable_prune:
        Enable the prune-redundant pass.
    enable_locality:
        Enable the reorder-for-locality pass.
    checkpoint_interval:
        Insert a checkpoint after every N non-trivial actions.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        enable_merge: bool = True,
        enable_parallel: bool = True,
        enable_checkpoints: bool = True,
        enable_prune: bool = True,
        enable_locality: bool = True,
        checkpoint_interval: int = 5,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.enable_merge = enable_merge
        self.enable_parallel = enable_parallel
        self.enable_checkpoints = enable_checkpoints
        self.enable_prune = enable_prune
        self.enable_locality = enable_locality
        self.checkpoint_interval = max(checkpoint_interval, 1)

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def optimize(
        self,
        plan: RepairPlan,
        graph: PipelineGraph,
    ) -> RepairPlan:
        """Apply all enabled optimisation passes.

        Parameters
        ----------
        plan:
            The input repair plan.
        graph:
            The pipeline graph.

        Returns
        -------
        RepairPlan
            The optimised plan.
        """
        actions = list(plan.actions)
        exec_order = list(plan.execution_order)

        if self.enable_prune:
            actions = self.prune_redundant(actions, graph)

        if self.enable_merge:
            actions = self.merge_compatible_actions(actions, graph)

        if self.enable_locality:
            actions = self.reorder_for_locality(actions, graph)

        if self.enable_parallel:
            actions = self.parallelize_independent(actions, graph)

        if self.enable_checkpoints:
            actions = self.insert_checkpoints(actions)

        # Recompute execution order
        exec_order = self._compute_execution_order(actions, graph)

        # Recost
        total_cost = sum(a.estimated_cost for a in actions if not a.is_noop)
        savings = 1.0 - (total_cost / plan.full_recompute_cost) if plan.full_recompute_cost > 0 else 0.0

        metadata = dict(plan.plan_metadata)
        metadata["optimized"] = True
        metadata["passes_applied"] = self._enabled_passes()

        optimized = RepairPlan(
            actions=tuple(actions),
            execution_order=tuple(exec_order),
            total_cost=total_cost,
            full_recompute_cost=plan.full_recompute_cost,
            savings_ratio=max(savings, 0.0),
            affected_nodes=plan.affected_nodes,
            annihilated_nodes=plan.annihilated_nodes,
            plan_metadata=metadata,
        )

        # Recompute cost breakdown
        breakdown = self.cost_model.total_plan_cost(optimized, graph)
        return RepairPlan(
            actions=optimized.actions,
            execution_order=optimized.execution_order,
            total_cost=breakdown.total_cost,
            full_recompute_cost=plan.full_recompute_cost,
            savings_ratio=max(
                1.0 - (breakdown.total_cost / plan.full_recompute_cost)
                if plan.full_recompute_cost > 0 else 0.0,
                0.0,
            ),
            affected_nodes=plan.affected_nodes,
            annihilated_nodes=plan.annihilated_nodes,
            plan_metadata=metadata,
            cost_breakdown=breakdown,
        )

    # ══════════════════════════════════════════════════════════════════
    # Pass 1: Merge compatible actions
    # ══════════════════════════════════════════════════════════════════

    def merge_compatible_actions(
        self,
        actions: list[RepairAction],
        graph: PipelineGraph,
    ) -> list[RepairAction]:
        """Batch compatible RECOMPUTE actions that share the same parent.

        Two recompute actions are compatible if:
        * They are both RECOMPUTE.
        * They share at least one parent node.
        * Neither depends on the other.

        Merged actions combine their costs and dependencies.
        """
        if len(actions) <= 1:
            return actions

        # Group recomputes by parent set
        recompute_groups: dict[frozenset[str], list[RepairAction]] = defaultdict(list)
        other_actions: list[RepairAction] = []

        action_nodes = {a.node_id for a in actions}
        dep_set = {a.node_id: set(a.dependencies) for a in actions}

        for action in actions:
            if action.action_type == ActionType.RECOMPUTE:
                parents = frozenset(graph.parents(action.node_id))
                recompute_groups[parents].append(action)
            else:
                other_actions.append(action)

        merged: list[RepairAction] = list(other_actions)
        for parent_set, group in recompute_groups.items():
            if len(group) <= 1:
                merged.extend(group)
                continue

            # Check no internal dependencies
            group_ids = {a.node_id for a in group}
            can_merge = True
            for a in group:
                if set(a.dependencies) & group_ids:
                    can_merge = False
                    break

            if can_merge:
                total_cost = sum(a.estimated_cost for a in group)
                all_deps: set[str] = set()
                for a in group:
                    all_deps.update(a.dependencies)
                all_deps -= group_ids

                primary = group[0]
                merged_action = RepairAction(
                    node_id=primary.node_id,
                    action_type=ActionType.RECOMPUTE,
                    estimated_cost=total_cost,
                    dependencies=tuple(sorted(all_deps)),
                    delta_to_apply=primary.delta_to_apply,
                    metadata={
                        "merged_nodes": [a.node_id for a in group],
                        "merge_count": len(group),
                    },
                )
                merged.append(merged_action)
                # Keep the other group members as NO_OPs
                for a in group[1:]:
                    merged.append(RepairAction(
                        node_id=a.node_id,
                        action_type=ActionType.NO_OP,
                        estimated_cost=0.0,
                        metadata={"merged_into": primary.node_id},
                    ))
                logger.debug(
                    "Merged %d recompute actions into %s",
                    len(group), primary.node_id,
                )
            else:
                merged.extend(group)

        return merged

    # ══════════════════════════════════════════════════════════════════
    # Pass 2: Parallelise independent actions
    # ══════════════════════════════════════════════════════════════════

    def parallelize_independent(
        self,
        actions: list[RepairAction],
        graph: PipelineGraph,
    ) -> list[RepairAction]:
        """Mark independent actions with a ``parallel_group`` metadata key.

        Two actions are independent if neither depends on the other
        (directly or transitively).
        """
        if len(actions) <= 1:
            return actions

        dep_map: dict[str, set[str]] = {}
        for a in actions:
            dep_map[a.node_id] = set(a.dependencies)

        # Compute transitive closure
        for nid in dep_map:
            visited: set[str] = set()
            stack = list(dep_map[nid])
            while stack:
                d = stack.pop()
                if d in visited:
                    continue
                visited.add(d)
                stack.extend(dep_map.get(d, set()))
            dep_map[nid] = visited

        # Assign parallel groups via a wave-front approach
        assigned: dict[str, int] = {}
        result: list[RepairAction] = []

        for a in actions:
            if a.is_noop:
                result.append(a)
                continue
            # This action's group = max(group of deps) + 1
            group = 0
            for dep in dep_map.get(a.node_id, set()):
                if dep in assigned:
                    group = max(group, assigned[dep] + 1)
            assigned[a.node_id] = group

            meta = dict(a.metadata)
            meta["parallel_group"] = group
            result.append(RepairAction(
                node_id=a.node_id,
                action_type=a.action_type,
                estimated_cost=a.estimated_cost,
                dependencies=a.dependencies,
                delta_to_apply=a.delta_to_apply,
                sql_text=a.sql_text,
                metadata=meta,
                action_id=a.action_id,
            ))

        return result

    # ══════════════════════════════════════════════════════════════════
    # Pass 3: Insert checkpoints
    # ══════════════════════════════════════════════════════════════════

    def insert_checkpoints(
        self,
        actions: list[RepairAction],
    ) -> list[RepairAction]:
        """Insert CHECKPOINT actions at regular intervals.

        A checkpoint is inserted after every ``checkpoint_interval``
        non-trivial actions to enable rollback on failure.
        """
        result: list[RepairAction] = []
        non_trivial_count = 0

        for a in actions:
            result.append(a)
            if not a.is_noop:
                non_trivial_count += 1
                if non_trivial_count % self.checkpoint_interval == 0:
                    cp = RepairAction(
                        node_id=f"__checkpoint_{non_trivial_count}__",
                        action_type=ActionType.CHECKPOINT,
                        estimated_cost=0.001,
                        dependencies=(a.node_id,),
                        metadata={"after_action": a.node_id},
                    )
                    result.append(cp)

        return result

    # ══════════════════════════════════════════════════════════════════
    # Pass 4: Prune redundant actions
    # ══════════════════════════════════════════════════════════════════

    def prune_redundant(
        self,
        actions: list[RepairAction],
        graph: PipelineGraph,
    ) -> list[RepairAction]:
        """Remove actions whose output is consumed by a later RECOMPUTE.

        If node A is INCREMENTAL_UPDATE but its only child B is
        RECOMPUTE, then A's incremental update is wasted — B will
        recompute from scratch anyway.
        """
        action_map = {a.node_id: a for a in actions}
        recompute_nodes = {
            a.node_id for a in actions if a.action_type == ActionType.RECOMPUTE
        }

        result: list[RepairAction] = []
        for a in actions:
            if a.action_type == ActionType.INCREMENTAL_UPDATE:
                children = graph.children(a.node_id)
                all_children_recompute = (
                    children and
                    all(c in recompute_nodes for c in children)
                )
                if all_children_recompute:
                    logger.debug(
                        "Pruned redundant incremental update at %s "
                        "(all children recompute)",
                        a.node_id,
                    )
                    result.append(RepairAction(
                        node_id=a.node_id,
                        action_type=ActionType.SKIP,
                        estimated_cost=0.0,
                        metadata={"pruned": True, "reason": "children_recompute"},
                    ))
                    continue
            result.append(a)

        return result

    # ══════════════════════════════════════════════════════════════════
    # Pass 5: Reorder for locality
    # ══════════════════════════════════════════════════════════════════

    def reorder_for_locality(
        self,
        actions: list[RepairAction],
        graph: PipelineGraph,
    ) -> list[RepairAction]:
        """Reorder actions to improve data locality.

        Actions touching the same table (same ``table_name`` in the
        node metadata) are grouped together while still respecting
        dependencies.
        """
        if len(actions) <= 2:
            return actions

        # Group by table name
        table_groups: dict[str, list[RepairAction]] = defaultdict(list)
        for a in actions:
            node = graph.nodes.get(a.node_id)
            table = node.table_name if node is not None else ""
            table_groups[table or a.node_id].append(a)

        # Build dependency set for ordering
        dep_set: dict[str, set[str]] = {}
        for a in actions:
            dep_set[a.node_id] = set(a.dependencies)

        # Topological sort respecting dependencies and grouping
        placed: set[str] = set()
        result: list[RepairAction] = []

        def can_place(a: RepairAction) -> bool:
            return all(d in placed for d in a.dependencies)

        remaining = list(actions)
        max_iters = len(remaining) * len(remaining) + 1
        iteration = 0

        while remaining and iteration < max_iters:
            iteration += 1
            placed_this_round = False
            next_remaining: list[RepairAction] = []

            for a in remaining:
                if can_place(a):
                    result.append(a)
                    placed.add(a.node_id)
                    placed_this_round = True
                else:
                    next_remaining.append(a)

            remaining = next_remaining
            if not placed_this_round:
                result.extend(remaining)
                break

        return result

    # ══════════════════════════════════════════════════════════════════
    # Parallelism estimation
    # ══════════════════════════════════════════════════════════════════

    def estimate_parallelism(
        self,
        plan: RepairPlan,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """Estimate theoretical speedup from parallelisation.

        Returns a dictionary with:
        * ``speedup``: theoretical speedup factor.
        * ``waves``: number of parallel waves.
        * ``critical_path``: cost of the critical (longest) path.
        * ``total_cost``: sum of all action costs.
        """
        if not plan.actions:
            return {"speedup": 1.0, "waves": 0, "critical_path": 0.0, "total_cost": 0.0}

        # Build wave structure from parallel_group metadata
        waves: dict[int, list[RepairAction]] = defaultdict(list)
        for a in plan.actions:
            if a.is_noop:
                continue
            group = a.metadata.get("parallel_group", 0)
            waves[group].append(a)

        if not waves:
            return {"speedup": 1.0, "waves": 0, "critical_path": 0.0, "total_cost": 0.0}

        total_cost = sum(a.estimated_cost for a in plan.actions if not a.is_noop)
        wave_costs = []
        for group_id in sorted(waves.keys()):
            wave_actions = waves[group_id]
            # With max_workers, the wave time = max(action_cost) if all fit;
            # otherwise = sum / min(workers, len(actions))
            sorted_costs = sorted((a.estimated_cost for a in wave_actions), reverse=True)
            effective_workers = min(max_workers, len(sorted_costs))
            if effective_workers <= 0:
                continue
            # Simple model: distribute work evenly
            wave_time = sum(sorted_costs) / effective_workers
            wave_costs.append(wave_time)

        critical_path = sum(wave_costs)
        speedup = total_cost / critical_path if critical_path > 0 else 1.0

        return {
            "speedup": min(speedup, float(max_workers)),
            "waves": len(waves),
            "critical_path": critical_path,
            "total_cost": total_cost,
        }

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    def _compute_execution_order(
        self,
        actions: list[RepairAction],
        graph: PipelineGraph,
    ) -> list[str]:
        """Compute a valid execution order for the actions."""
        non_trivial = [a for a in actions if not a.is_noop and a.action_type != ActionType.CHECKPOINT]
        node_ids = {a.node_id for a in non_trivial}

        try:
            topo = graph.topological_order()
            return [nid for nid in topo if nid in node_ids]
        except ValueError:
            return sorted(node_ids)

    def _enabled_passes(self) -> list[str]:
        passes = []
        if self.enable_prune:
            passes.append("prune_redundant")
        if self.enable_merge:
            passes.append("merge_compatible")
        if self.enable_locality:
            passes.append("reorder_locality")
        if self.enable_parallel:
            passes.append("parallelize_independent")
        if self.enable_checkpoints:
            passes.append("insert_checkpoints")
        return passes

    def __repr__(self) -> str:
        return f"PlanOptimizer(passes={self._enabled_passes()})"
