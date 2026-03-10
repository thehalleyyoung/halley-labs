"""
LP-relaxation repair planner for general topologies (Algorithm A4).

Uses linear-programming relaxation with randomised rounding to produce
a (ln k + 1)-approximation repair plan for pipeline graphs that may
contain cycles (e.g. feedback loops in ML training pipelines).

Algorithm
---------
1. Formulate a 0/1 Integer LP:
   * Variable x_v ∈ {0, 1} for each affected node v (1 = repair).
   * Objective: minimise Σ cost(v) · x_v.
   * Coverage: every affected node must be repaired or be downstream
     of a repaired node.
   * Dependency: if propagation is needed on edge (u, v), then
     x_v ≥ 1 − x_u.
2. Solve the LP relaxation (x_v ∈ [0, 1]) via ``scipy.optimize.linprog``.
3. Randomised rounding: round x_v ≥ 0.5 to 1; probabilistically round
   others.
4. Greedy feasibility patch: ensure all coverage constraints are met.
5. Local search: try removing each repair action; keep removal if the
   plan remains feasible.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np

from arc.planner.cost import CostModel
from arc.types.base import (
    ActionType,
    CompoundPerturbation,
    CostBreakdown,
    PipelineGraph,
    PipelineNode,
    RepairAction,
    RepairPlan,
    SQLOperator,
)

logger = logging.getLogger(__name__)


class LPRepairPlanner:
    """(ln k + 1)-approximation repair planner for general topologies.

    Parameters
    ----------
    cost_model:
        Cost model.  A default is used when *None*.
    seed:
        Random seed for the rounding phase.
    rounding_threshold:
        LP value above which a variable is deterministically rounded to 1.
    max_local_search_iterations:
        Maximum iterations for the local-search improvement phase.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        seed: int = 42,
        rounding_threshold: float = 0.5,
        max_local_search_iterations: int = 100,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.seed = seed
        self.rounding_threshold = rounding_threshold
        self.max_local_search_iters = max_local_search_iterations

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def plan(
        self,
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
        cost_model: CostModel | None = None,
    ) -> RepairPlan:
        """Compute an approximately optimal repair plan.

        Parameters
        ----------
        graph:
            The pipeline graph (may contain cycles).
        deltas:
            Source perturbations keyed by node ID.
        cost_model:
            Optional override cost model.

        Returns
        -------
        RepairPlan
        """
        cm = cost_model or self.cost_model

        if not deltas:
            return self._empty_plan(graph, cm)

        # 1. Identify affected nodes
        affected = self._identify_affected_nodes(graph, deltas)
        if not affected:
            return self._empty_plan(graph, cm)

        # Index nodes
        node_list = sorted(affected)
        node_idx = {nid: i for i, nid in enumerate(node_list)}
        n = len(node_list)

        # 2. Compute costs
        costs = np.zeros(n)
        for i, nid in enumerate(node_list):
            node = graph.nodes[nid]
            costs[i] = cm.estimate_recompute_cost(
                node,
                self._parent_sizes(graph, nid),
            )
        # Normalise to avoid numerical issues
        cost_scale = max(costs.max(), 1e-12)
        scaled_costs = costs / cost_scale

        # 3. Formulate LP
        # Constraints: A_ub @ x <= b_ub  (we negate for >= constraints)
        A_rows: list[list[float]] = []
        b_vals: list[float] = []

        # Coverage constraints: each affected node v needs
        #   x_v + Σ_{u→v, u in affected} x_u >= 1
        # Rewritten as: -(x_v + Σ x_u) <= -1
        for nid in node_list:
            row = [0.0] * n
            idx = node_idx[nid]
            row[idx] = -1.0
            # Parents that are also affected
            for pid in graph.parents(nid):
                if pid in node_idx:
                    row[node_idx[pid]] = -1.0
            A_rows.append(row)
            b_vals.append(-1.0)

        # Dependency constraints: if edge (u, v) and both affected,
        #   x_v >= 1 - x_u  =>  x_u + x_v >= 1  =>  -(x_u + x_v) <= -1
        for nid in node_list:
            for pid in graph.parents(nid):
                if pid in node_idx:
                    row = [0.0] * n
                    row[node_idx[pid]] = -1.0
                    row[node_idx[nid]] = -1.0
                    A_rows.append(row)
                    b_vals.append(-1.0)

        A_ub = np.array(A_rows) if A_rows else np.zeros((0, n))
        b_ub = np.array(b_vals) if b_vals else np.zeros(0)
        bounds = [(0.0, 1.0)] * n

        # 4. Solve LP relaxation
        lp_solution = self._solve_lp(scaled_costs, A_ub, b_ub, bounds, n)

        # 5. Randomised rounding
        rng = random.Random(self.seed)
        rounded = self._randomised_rounding(lp_solution, rng)

        # 6. Greedy feasibility patch
        rounded = self._greedy_feasibility_patch(
            rounded, A_ub, b_ub, scaled_costs, node_list, node_idx, graph
        )

        # 7. Local search improvement
        rounded = self._local_search(
            rounded, A_ub, b_ub, scaled_costs, node_list, node_idx, graph
        )

        # 8. Build repair plan
        plan = self._build_plan(
            rounded, node_list, node_idx, graph, deltas, costs, cm
        )

        return plan

    # ══════════════════════════════════════════════════════════════════
    # LP solver
    # ══════════════════════════════════════════════════════════════════

    def _solve_lp(
        self,
        c: np.ndarray,
        A_ub: np.ndarray,
        b_ub: np.ndarray,
        bounds: list[tuple[float, float]],
        n: int,
    ) -> np.ndarray:
        """Solve the LP relaxation using scipy.optimize.linprog.

        Falls back to a heuristic (all-1) solution if the LP solver
        fails or is unavailable.
        """
        try:
            from scipy.optimize import linprog, LinearConstraint

            result = linprog(
                c=c,
                A_ub=A_ub if A_ub.size > 0 else None,
                b_ub=b_ub if b_ub.size > 0 else None,
                bounds=bounds,
                method="highs",
                options={"maxiter": 10000, "disp": False},
            )
            if result.success:
                logger.info(
                    "LP relaxation solved: obj=%.6f, status=%s",
                    result.fun, result.message,
                )
                return np.clip(result.x, 0.0, 1.0)
            else:
                logger.warning(
                    "LP solver did not converge: %s. Falling back to heuristic.",
                    result.message,
                )
                return np.ones(n)
        except ImportError:
            logger.warning("scipy not available; using heuristic solution.")
            return np.ones(n)
        except Exception as exc:
            logger.warning("LP solver error: %s. Using heuristic.", exc)
            return np.ones(n)

    # ══════════════════════════════════════════════════════════════════
    # Randomised rounding
    # ══════════════════════════════════════════════════════════════════

    def _randomised_rounding(
        self,
        lp_values: np.ndarray,
        rng: random.Random,
    ) -> np.ndarray:
        """Round LP solution to integer values.

        * Values ≥ ``rounding_threshold`` are rounded to 1.
        * Others are rounded to 1 with probability equal to their LP value.
        """
        n = len(lp_values)
        rounded = np.zeros(n)
        for i in range(n):
            if lp_values[i] >= self.rounding_threshold:
                rounded[i] = 1.0
            elif rng.random() < lp_values[i]:
                rounded[i] = 1.0
        return rounded

    # ══════════════════════════════════════════════════════════════════
    # Greedy feasibility patch
    # ══════════════════════════════════════════════════════════════════

    def _greedy_feasibility_patch(
        self,
        x: np.ndarray,
        A_ub: np.ndarray,
        b_ub: np.ndarray,
        costs: np.ndarray,
        node_list: list[str],
        node_idx: dict[str, int],
        graph: PipelineGraph,
    ) -> np.ndarray:
        """Greedily add cheapest repairs to satisfy all constraints.

        After rounding, some coverage constraints may be violated.
        This pass adds the cheapest missing repairs until all
        constraints are satisfied.
        """
        x = x.copy()

        if A_ub.size == 0:
            return x

        max_iters = len(node_list) * 2
        for _ in range(max_iters):
            violations = A_ub @ x - b_ub
            violated_indices = np.where(violations > 1e-9)[0]
            if len(violated_indices) == 0:
                break

            # Find the cheapest node to add that fixes the most violations
            best_node = -1
            best_score = float("inf")

            for j in range(len(node_list)):
                if x[j] >= 1.0:
                    continue
                # How many constraints would this fix?
                x_test = x.copy()
                x_test[j] = 1.0
                new_violations = A_ub @ x_test - b_ub
                fixes = np.sum(violations > 1e-9) - np.sum(new_violations > 1e-9)
                if fixes > 0:
                    score = costs[j] / max(fixes, 1)
                    if score < best_score:
                        best_score = score
                        best_node = j

            if best_node < 0:
                # No single node fixes violations — add all violated nodes
                for vi in violated_indices:
                    row = A_ub[vi]
                    for j in range(len(node_list)):
                        if row[j] < -0.5 and x[j] < 1.0:
                            x[j] = 1.0
                            break
                break

            x[best_node] = 1.0
            logger.debug(
                "Feasibility patch: added node %s (cost %.6f)",
                node_list[best_node], costs[best_node],
            )

        return x

    # ══════════════════════════════════════════════════════════════════
    # Local search improvement
    # ══════════════════════════════════════════════════════════════════

    def _local_search(
        self,
        x: np.ndarray,
        A_ub: np.ndarray,
        b_ub: np.ndarray,
        costs: np.ndarray,
        node_list: list[str],
        node_idx: dict[str, int],
        graph: PipelineGraph,
    ) -> np.ndarray:
        """Try removing each repair action; keep removal if still feasible.

        Iterates in decreasing cost order (try removing expensive
        repairs first).
        """
        x = x.copy()
        order = sorted(
            range(len(node_list)),
            key=lambda i: -costs[i],
        )

        improved = True
        iterations = 0
        while improved and iterations < self.max_local_search_iters:
            improved = False
            iterations += 1

            for j in order:
                if x[j] < 1.0:
                    continue
                # Try removing
                x_test = x.copy()
                x_test[j] = 0.0

                if A_ub.size > 0:
                    violations = A_ub @ x_test - b_ub
                    if np.any(violations > 1e-9):
                        continue

                x[j] = 0.0
                improved = True
                logger.debug(
                    "Local search: removed node %s (saved %.6f)",
                    node_list[j], costs[j],
                )

        return x

    # ══════════════════════════════════════════════════════════════════
    # Plan construction
    # ══════════════════════════════════════════════════════════════════

    def _build_plan(
        self,
        x: np.ndarray,
        node_list: list[str],
        node_idx: dict[str, int],
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
        costs: np.ndarray,
        cost_model: CostModel,
    ) -> RepairPlan:
        """Build a RepairPlan from the rounded LP solution."""
        actions: list[RepairAction] = []
        repair_nodes: set[str] = set()

        for i, nid in enumerate(node_list):
            if x[i] >= 0.5:
                node = graph.nodes[nid]
                delta = deltas.get(nid, CompoundPerturbation())
                action_type = cost_model.choose_action_type(
                    node, delta, self._parent_sizes(graph, nid)
                )
                actions.append(RepairAction(
                    node_id=nid,
                    action_type=action_type,
                    estimated_cost=float(costs[i]),
                    dependencies=tuple(
                        pid for pid in graph.parents(nid)
                        if pid in repair_nodes
                    ),
                    delta_to_apply=delta if not delta.is_identity else None,
                ))
                repair_nodes.add(nid)
            else:
                actions.append(RepairAction(
                    node_id=nid,
                    action_type=ActionType.SKIP,
                    estimated_cost=0.0,
                ))

        # Execution order: try topological; fall back to lexicographic
        try:
            topo = graph.topological_order()
            exec_order = [nid for nid in topo if nid in repair_nodes]
        except ValueError:
            exec_order = sorted(repair_nodes)

        total_cost = sum(a.estimated_cost for a in actions if not a.is_noop)
        full_cost = cost_model.estimate_full_recompute_cost(graph)
        savings = 1.0 - (total_cost / full_cost) if full_cost > 0 else 0.0

        affected = frozenset(node_list)
        skipped = frozenset(nid for nid, xi in zip(node_list, x) if xi < 0.5)

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
            affected_nodes=affected,
            annihilated_nodes=frozenset(),
            plan_metadata={
                "planner": "LPRepairPlanner",
                "algorithm": "A4",
                "topology": "general",
                "lp_nodes": len(node_list),
                "repaired_nodes": len(repair_nodes),
                "skipped_nodes": len(skipped),
                "approximation_ratio": f"ln({len(affected)})+1",
            },
            cost_breakdown=cost_breakdown,
        )

    # ══════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════

    def _identify_affected_nodes(
        self,
        graph: PipelineGraph,
        deltas: dict[str, CompoundPerturbation],
    ) -> set[str]:
        """Identify all nodes affected by the given deltas.

        A node is affected if it has a direct delta or is reachable
        from a node with a delta.
        """
        affected: set[str] = set()
        for nid in deltas:
            if nid in graph.nodes and not deltas[nid].is_identity:
                affected |= graph.reachable_from(nid)
        return affected & set(graph.nodes.keys())

    def _parent_sizes(
        self,
        graph: PipelineGraph,
        node_id: str,
    ) -> dict[str, int]:
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
        full_cost = cost_model.estimate_full_recompute_cost(graph)
        return RepairPlan(
            full_recompute_cost=full_cost,
            savings_ratio=1.0,
            plan_metadata={
                "planner": "LPRepairPlanner",
                "algorithm": "A4",
                "topology": "general",
                "note": "no deltas to repair",
            },
        )

    def __repr__(self) -> str:
        return (
            f"LPRepairPlanner(threshold={self.rounding_threshold}, "
            f"seed={self.seed})"
        )
