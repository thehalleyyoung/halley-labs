"""
Execution scheduler for repair plans.

:class:`ExecutionScheduler` organises repair actions into *waves* of
independent actions that can execute in parallel, while respecting
inter-action dependencies and resource constraints.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from arc.planner.cost import CostModel
from arc.types.base import (
    ActionType,
    ExecutionSchedule,
    PipelineGraph,
    RepairAction,
    RepairPlan,
    ResourceSpec,
)

logger = logging.getLogger(__name__)


class ExecutionScheduler:
    """Schedule repair actions respecting dependencies and resources.

    Parameters
    ----------
    cost_model:
        Cost model for time estimation.
    default_max_parallelism:
        Default maximum number of parallel workers.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        default_max_parallelism: int = 4,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.default_max_parallelism = default_max_parallelism

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def schedule(
        self,
        plan: RepairPlan,
        max_parallelism: int | None = None,
    ) -> ExecutionSchedule:
        """Schedule repair actions into parallel waves.

        Parameters
        ----------
        plan:
            The repair plan to schedule.
        max_parallelism:
            Maximum number of actions per wave.

        Returns
        -------
        ExecutionSchedule
        """
        max_p = max_parallelism or self.default_max_parallelism
        waves = self.topological_schedule(plan)

        # Split waves that exceed max_parallelism
        bounded_waves: list[tuple[RepairAction, ...]] = []
        for wave in waves:
            if len(wave) <= max_p:
                bounded_waves.append(tuple(wave))
            else:
                for i in range(0, len(wave), max_p):
                    bounded_waves.append(tuple(wave[i:i + max_p]))

        # Estimate times
        total_time = sum(a.estimated_cost for a in plan.actions if not a.is_noop)
        wave_times = []
        for wave in bounded_waves:
            costs = [a.estimated_cost for a in wave if not a.is_noop]
            wave_times.append(max(costs) if costs else 0.0)
        wall_time = sum(wave_times)

        crit_path = self._compute_critical_path_cost(plan)
        par_factor = total_time / wall_time if wall_time > 0 else 1.0

        return ExecutionSchedule(
            waves=tuple(bounded_waves),
            estimated_wall_time=wall_time,
            estimated_total_time=total_time,
            critical_path_length=crit_path,
            parallelism_factor=par_factor,
        )

    def topological_schedule(
        self,
        plan: RepairPlan,
    ) -> list[list[RepairAction]]:
        """Organise actions into waves based on dependencies.

        Each wave contains actions whose dependencies are all satisfied
        by previous waves.  Actions within a wave are independent.
        """
        # Build dependency graph
        actions = [a for a in plan.actions if not a.is_noop]
        action_map = {a.node_id: a for a in actions}
        remaining = set(a.node_id for a in actions)
        completed: set[str] = set()
        waves: list[list[RepairAction]] = []

        max_iters = len(actions) + 1
        iteration = 0

        while remaining and iteration < max_iters:
            iteration += 1
            wave: list[RepairAction] = []

            for nid in list(remaining):
                action = action_map[nid]
                deps = set(action.dependencies) & set(action_map.keys())
                if deps <= completed:
                    wave.append(action)

            if not wave:
                # Circular dependency or no progress — dump remaining
                for nid in sorted(remaining):
                    wave.append(action_map[nid])
                waves.append(wave)
                break

            waves.append(wave)
            for a in wave:
                remaining.discard(a.node_id)
                completed.add(a.node_id)

        return waves

    def critical_path(self, plan: RepairPlan) -> list[RepairAction]:
        """Compute the critical (longest) path through the plan.

        The critical path is the sequence of dependent actions with the
        highest total cost.
        """
        actions = [a for a in plan.actions if not a.is_noop]
        action_map = {a.node_id: a for a in actions}

        # DP for longest path
        memo: dict[str, float] = {}
        parent_of: dict[str, str] = {}

        def longest(nid: str) -> float:
            if nid in memo:
                return memo[nid]
            action = action_map.get(nid)
            if action is None:
                memo[nid] = 0.0
                return 0.0

            best = 0.0
            best_parent = ""
            for dep in action.dependencies:
                if dep in action_map:
                    dep_cost = longest(dep)
                    if dep_cost > best:
                        best = dep_cost
                        best_parent = dep

            cost = best + action.estimated_cost
            memo[nid] = cost
            if best_parent:
                parent_of[nid] = best_parent
            return cost

        for nid in action_map:
            longest(nid)

        if not memo:
            return []

        # Backtrack from the node with highest cost
        end_node = max(memo, key=lambda k: memo[k])
        path: list[RepairAction] = []
        current: str | None = end_node

        while current is not None and current in action_map:
            path.append(action_map[current])
            current = parent_of.get(current)

        path.reverse()
        return path

    def estimate_wall_time(
        self,
        schedule: ExecutionSchedule,
        cost_model: CostModel | None = None,
    ) -> float:
        """Estimate wall-clock time for an execution schedule.

        The wall time is the sum of wave times, where each wave's time
        is the maximum action cost in that wave.
        """
        total = 0.0
        for wave in schedule.waves:
            costs = [a.estimated_cost for a in wave if not a.is_noop]
            total += max(costs) if costs else 0.0
        return total

    def resource_constrained_schedule(
        self,
        plan: RepairPlan,
        resources: ResourceSpec,
    ) -> ExecutionSchedule:
        """Schedule with resource constraints.

        Respects ``max_parallelism`` and optionally ``timeout_seconds``.
        """
        schedule = self.schedule(plan, max_parallelism=resources.max_parallelism)

        if resources.timeout_seconds > 0:
            # Truncate waves that would exceed the timeout
            truncated_waves: list[tuple[RepairAction, ...]] = []
            elapsed = 0.0
            for wave in schedule.waves:
                costs = [a.estimated_cost for a in wave if not a.is_noop]
                wave_time = max(costs) if costs else 0.0
                if elapsed + wave_time > resources.timeout_seconds:
                    break
                truncated_waves.append(wave)
                elapsed += wave_time

            return ExecutionSchedule(
                waves=tuple(truncated_waves),
                estimated_wall_time=elapsed,
                estimated_total_time=sum(
                    a.estimated_cost for w in truncated_waves for a in w if not a.is_noop
                ),
                critical_path_length=schedule.critical_path_length,
                parallelism_factor=schedule.parallelism_factor,
            )

        return schedule

    # ── Private helpers ────────────────────────────────────────────────

    def _compute_critical_path_cost(self, plan: RepairPlan) -> float:
        """Compute the cost of the critical path."""
        cp = self.critical_path(plan)
        return sum(a.estimated_cost for a in cp)

    def __repr__(self) -> str:
        return f"ExecutionScheduler(max_parallel={self.default_max_parallelism})"
