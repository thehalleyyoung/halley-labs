"""
``arc.planner`` — Repair planning for the Algebraic Repair Calculus.

Provides cost-model-based repair planners:

* :class:`CostModel` – operator-aware cost estimation.
* :class:`DPRepairPlanner` – optimal DP planner for acyclic DAGs (Algorithm A3).
* :class:`LPRepairPlanner` – LP-relaxation approximation for general graphs (Algorithm A4).
* :class:`PlanOptimizer` – post-optimization passes on repair plans.
"""

from arc.planner.cost import CostModel, CostFactors
from arc.planner.dp import DPRepairPlanner
from arc.planner.lp import LPRepairPlanner
from arc.planner.optimizer import PlanOptimizer
from arc.planner.strategy import (
    PipelineAnalyzer,
    PipelineCharacteristics,
    PipelineScale,
    PlannerType,
    PerturbationType,
    RepairStrategy,
    RepairStrategySelector,
    StrategyComparison,
    compare_strategies,
    select_strategy,
)
from arc.planner.greedy import (
    GreedyHeuristic,
    GreedyRepairAction,
    GreedyRepairPlan,
    GreedyRepairPlanner,
    RepairCandidate,
    greedy_plan,
    greedy_plan_with_budget,
)

__all__ = [
    "CostModel",
    "CostFactors",
    "DPRepairPlanner",
    "LPRepairPlanner",
    "PlanOptimizer",
    # Strategy
    "PipelineAnalyzer",
    "PipelineCharacteristics",
    "PipelineScale",
    "PlannerType",
    "PerturbationType",
    "RepairStrategy",
    "RepairStrategySelector",
    "StrategyComparison",
    "compare_strategies",
    "select_strategy",
    # Greedy
    "GreedyHeuristic",
    "GreedyRepairAction",
    "GreedyRepairPlan",
    "GreedyRepairPlanner",
    "RepairCandidate",
    "greedy_plan",
    "greedy_plan_with_budget",
]
