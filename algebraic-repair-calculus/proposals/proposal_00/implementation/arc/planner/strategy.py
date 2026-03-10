"""
Repair Strategy Selection
==========================

Selects the optimal repair strategy based on pipeline characteristics,
perturbation type, scale, time budget, and resource constraints.

Available strategies:
  - **DP** (Dynamic Programming): Optimal for acyclic graphs.
  - **LP** (Linear Programming): (ln k + 1)-approximation for general graphs.
  - **Greedy**: Fast heuristic for real-time scenarios.
  - **Hybrid**: Start with greedy, improve with DP/LP if time allows.
  - **Adaptive**: Progressively improve the plan within a time budget.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from arc.planner.cost import CostModel

logger = logging.getLogger(__name__)


# =====================================================================
# Strategy Types
# =====================================================================


class PlannerType(Enum):
    """Available repair planner types."""
    DP = "dp"
    LP = "lp"
    GREEDY = "greedy"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    NOOP = "noop"


class PerturbationType(Enum):
    """Classification of perturbation types."""
    SCHEMA_ONLY = "schema_only"
    DATA_ONLY = "data_only"
    QUALITY_ONLY = "quality_only"
    COMPOUND = "compound"
    UNKNOWN = "unknown"


class PipelineScale(Enum):
    """Classification of pipeline scale."""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


# =====================================================================
# Strategy and Comparison Results
# =====================================================================


@dataclass
class RepairStrategy:
    """A selected repair strategy with its rationale.

    Attributes
    ----------
    planner_type : PlannerType
        The selected planner.
    estimated_planning_time : float
        Estimated time to generate the plan (seconds).
    estimated_plan_quality : float
        Estimated quality of the plan (0.0 to 1.0, where 1.0 is optimal).
    estimated_execution_time : float
        Estimated time to execute the resulting plan (seconds).
    parameters : dict
        Planner-specific configuration parameters.
    rationale : str
        Human-readable explanation of why this strategy was selected.
    confidence : float
        Confidence in this selection (0.0 to 1.0).
    """
    planner_type: PlannerType = PlannerType.GREEDY
    estimated_planning_time: float = 0.0
    estimated_plan_quality: float = 0.5
    estimated_execution_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.5

    def summary(self) -> str:
        return (
            f"Strategy({self.planner_type.value}): "
            f"plan_time≈{self.estimated_planning_time:.2f}s, "
            f"quality≈{self.estimated_plan_quality:.2f}, "
            f"exec_time≈{self.estimated_execution_time:.2f}s "
            f"[confidence={self.confidence:.2f}]"
        )


@dataclass
class StrategyComparison:
    """Comparison of multiple repair strategies.

    Attributes
    ----------
    strategies : dict[str, RepairStrategy]
        All evaluated strategies.
    recommended : str
        The recommended strategy name.
    comparison_time : float
        Time spent comparing strategies (seconds).
    plans : dict[str, Any]
        Generated plans for each strategy (when available).
    metrics : dict[str, dict[str, float]]
        Detailed metrics for each strategy.
    """
    strategies: Dict[str, RepairStrategy] = field(default_factory=dict)
    recommended: str = ""
    comparison_time: float = 0.0
    plans: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"StrategyComparison (recommended: {self.recommended}):"]
        for name, strategy in self.strategies.items():
            marker = " ✓" if name == self.recommended else ""
            lines.append(f"  {name}{marker}: {strategy.summary()}")
        lines.append(f"  Comparison time: {self.comparison_time:.3f}s")
        return "\n".join(lines)


@dataclass
class PipelineCharacteristics:
    """Characteristics of a pipeline that influence strategy selection.

    Attributes
    ----------
    node_count : int
        Number of nodes in the pipeline.
    edge_count : int
        Number of edges.
    is_dag : bool
        Whether the pipeline is acyclic.
    max_depth : int
        Maximum topological depth.
    max_width : int
        Maximum width at any depth level.
    has_cycles : bool
        Whether the pipeline has feedback cycles.
    affected_node_count : int
        Number of nodes affected by the perturbation.
    perturbation_type : PerturbationType
        Type of perturbation.
    scale : PipelineScale
        Scale classification.
    density : float
        Edge density (edges / max_possible_edges).
    fan_out_max : int
        Maximum fan-out of any node.
    estimated_total_cost : float
        Estimated total repair cost.
    """
    node_count: int = 0
    edge_count: int = 0
    is_dag: bool = True
    max_depth: int = 0
    max_width: int = 0
    has_cycles: bool = False
    affected_node_count: int = 0
    perturbation_type: PerturbationType = PerturbationType.UNKNOWN
    scale: PipelineScale = PipelineScale.SMALL
    density: float = 0.0
    fan_out_max: int = 0
    estimated_total_cost: float = 0.0


# =====================================================================
# Pipeline Analyzer
# =====================================================================


class PipelineAnalyzer:
    """Analyze pipeline characteristics for strategy selection."""

    def analyze(
        self,
        graph: Any,
        perturbation: Any = None,
        affected_nodes: Optional[Set[str]] = None,
    ) -> PipelineCharacteristics:
        """Analyze a pipeline graph and perturbation.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph.
        perturbation : CompoundPerturbation, optional
            The perturbation.
        affected_nodes : set[str], optional
            Pre-computed affected nodes.

        Returns
        -------
        PipelineCharacteristics
        """
        n = graph.node_count
        e = graph.edge_count
        is_dag = graph.is_dag()

        max_depth = 0
        max_width = 0
        depth_map: Dict[int, int] = defaultdict(int)

        if is_dag:
            topo = graph.topological_sort()
            node_depths: Dict[str, int] = {}

            for nid in topo:
                preds = graph.predecessors(nid)
                if not preds:
                    node_depths[nid] = 0
                else:
                    node_depths[nid] = max(
                        node_depths.get(p, 0) for p in preds
                    ) + 1
                depth_map[node_depths[nid]] += 1

            max_depth = max(node_depths.values()) if node_depths else 0
            max_width = max(depth_map.values()) if depth_map else 0

        affected_count = len(affected_nodes) if affected_nodes else n
        pert_type = self._classify_perturbation(perturbation)
        scale = self._classify_scale(n, e, affected_count)

        max_possible_edges = n * (n - 1) / 2 if n > 1 else 1
        density = e / max_possible_edges if max_possible_edges > 0 else 0.0

        fan_out_max = 0
        for nid in graph.node_ids:
            out_deg = graph.out_degree(nid)
            fan_out_max = max(fan_out_max, out_deg)

        total_cost = 0.0
        for nid in graph.node_ids:
            node = graph.get_node(nid)
            total_cost += node.cost_estimate.total_weighted_cost

        return PipelineCharacteristics(
            node_count=n,
            edge_count=e,
            is_dag=is_dag,
            max_depth=max_depth,
            max_width=max_width,
            has_cycles=not is_dag,
            affected_node_count=affected_count,
            perturbation_type=pert_type,
            scale=scale,
            density=density,
            fan_out_max=fan_out_max,
            estimated_total_cost=total_cost,
        )

    @staticmethod
    def _classify_perturbation(perturbation: Any) -> PerturbationType:
        """Classify the type of perturbation."""
        if perturbation is None:
            return PerturbationType.UNKNOWN

        has_schema = (
            hasattr(perturbation, "schema_delta")
            and perturbation.schema_delta is not None
            and len(perturbation.schema_delta.operations) > 0
        )
        has_data = (
            hasattr(perturbation, "data_delta")
            and perturbation.data_delta is not None
            and len(perturbation.data_delta.operations) > 0
        )
        has_quality = (
            hasattr(perturbation, "quality_delta")
            and perturbation.quality_delta is not None
            and len(perturbation.quality_delta.operations) > 0
        )

        count = sum([has_schema, has_data, has_quality])

        if count == 0:
            return PerturbationType.UNKNOWN
        if count > 1:
            return PerturbationType.COMPOUND
        if has_schema:
            return PerturbationType.SCHEMA_ONLY
        if has_data:
            return PerturbationType.DATA_ONLY
        return PerturbationType.QUALITY_ONLY

    @staticmethod
    def _classify_scale(n: int, e: int, affected: int) -> PipelineScale:
        """Classify the scale of the pipeline."""
        total = n + e + affected
        if total <= 10:
            return PipelineScale.TINY
        if total <= 50:
            return PipelineScale.SMALL
        if total <= 500:
            return PipelineScale.MEDIUM
        if total <= 5000:
            return PipelineScale.LARGE
        return PipelineScale.VERY_LARGE


# =====================================================================
# Strategy Selector
# =====================================================================


class RepairStrategySelector:
    """Select the optimal repair strategy based on pipeline characteristics.

    Implements a rule-based strategy selection algorithm that considers:
    - Pipeline topology (acyclic → DP, cyclic → LP)
    - Perturbation type (schema-only, data-only, compound)
    - Scale (small → exact, large → approximate)
    - Time budget constraints
    - Resource constraints

    Parameters
    ----------
    cost_model : CostModel, optional
        Cost model for estimation.
    default_time_budget : float
        Default time budget for planning (seconds).
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        default_time_budget: float = 30.0,
    ) -> None:
        self._cost_model = cost_model or CostModel()
        self._default_budget = default_time_budget
        self._analyzer = PipelineAnalyzer()

    def select_strategy(
        self,
        graph: Any,
        perturbation: Any = None,
        constraints: Optional[Dict[str, Any]] = None,
        affected_nodes: Optional[Set[str]] = None,
    ) -> RepairStrategy:
        """Select the optimal repair strategy.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph.
        perturbation : CompoundPerturbation, optional
            The perturbation.
        constraints : dict, optional
            Planning constraints (time_budget, max_memory, etc.).
        affected_nodes : set[str], optional
            Pre-computed affected nodes.

        Returns
        -------
        RepairStrategy
        """
        chars = self._analyzer.analyze(graph, perturbation, affected_nodes)
        user_constraints = constraints or {}
        time_budget = user_constraints.get("time_budget", self._default_budget)

        if chars.affected_node_count == 0:
            return RepairStrategy(
                planner_type=PlannerType.NOOP,
                estimated_planning_time=0.0,
                estimated_plan_quality=1.0,
                rationale="No affected nodes — no repair needed",
                confidence=1.0,
            )

        if chars.has_cycles:
            return self._select_for_cyclic(chars, time_budget)

        if chars.scale in (PipelineScale.TINY, PipelineScale.SMALL):
            return self._select_for_small(chars, time_budget)

        if chars.scale == PipelineScale.MEDIUM:
            return self._select_for_medium(chars, time_budget)

        return self._select_for_large(chars, time_budget)

    def compare_strategies(
        self,
        graph: Any,
        perturbation: Any = None,
        affected_nodes: Optional[Set[str]] = None,
    ) -> StrategyComparison:
        """Compare multiple repair strategies.

        Evaluates each strategy and returns a comparison with metrics.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph.
        perturbation : CompoundPerturbation, optional
            The perturbation.
        affected_nodes : set[str], optional
            Pre-computed affected nodes.

        Returns
        -------
        StrategyComparison
        """
        start_time = time.time()
        chars = self._analyzer.analyze(graph, perturbation, affected_nodes)

        strategies: Dict[str, RepairStrategy] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        dp_strategy = self._evaluate_dp(chars)
        strategies["dp"] = dp_strategy
        metrics["dp"] = {
            "planning_time": dp_strategy.estimated_planning_time,
            "plan_quality": dp_strategy.estimated_plan_quality,
            "execution_time": dp_strategy.estimated_execution_time,
        }

        lp_strategy = self._evaluate_lp(chars)
        strategies["lp"] = lp_strategy
        metrics["lp"] = {
            "planning_time": lp_strategy.estimated_planning_time,
            "plan_quality": lp_strategy.estimated_plan_quality,
            "execution_time": lp_strategy.estimated_execution_time,
        }

        greedy_strategy = self._evaluate_greedy(chars)
        strategies["greedy"] = greedy_strategy
        metrics["greedy"] = {
            "planning_time": greedy_strategy.estimated_planning_time,
            "plan_quality": greedy_strategy.estimated_plan_quality,
            "execution_time": greedy_strategy.estimated_execution_time,
        }

        hybrid_strategy = self._evaluate_hybrid(chars)
        strategies["hybrid"] = hybrid_strategy
        metrics["hybrid"] = {
            "planning_time": hybrid_strategy.estimated_planning_time,
            "plan_quality": hybrid_strategy.estimated_plan_quality,
            "execution_time": hybrid_strategy.estimated_execution_time,
        }

        best_name = self._rank_strategies(strategies)

        comparison_time = time.time() - start_time

        return StrategyComparison(
            strategies=strategies,
            recommended=best_name,
            comparison_time=comparison_time,
            metrics=metrics,
        )

    def adaptive_planning(
        self,
        graph: Any,
        perturbation: Any = None,
        time_budget: float = 10.0,
        affected_nodes: Optional[Set[str]] = None,
    ) -> Tuple[RepairStrategy, Any]:
        """Adaptively plan within a time budget.

        Starts with a fast greedy solution, then tries to improve with
        DP or LP if time allows.

        Parameters
        ----------
        graph : PipelineGraph
            The pipeline graph.
        perturbation : CompoundPerturbation, optional
            The perturbation.
        time_budget : float
            Maximum planning time in seconds.
        affected_nodes : set[str], optional
            Pre-computed affected nodes.

        Returns
        -------
        tuple[RepairStrategy, Any]
            The selected strategy and generated plan (if available).
        """
        start_time = time.time()
        chars = self._analyzer.analyze(graph, perturbation, affected_nodes)

        greedy = self._evaluate_greedy(chars)
        greedy_plan = None

        elapsed = time.time() - start_time
        remaining = time_budget - elapsed

        if remaining <= 0.5:
            return greedy, greedy_plan

        if chars.is_dag and chars.scale in (PipelineScale.TINY, PipelineScale.SMALL):
            dp = self._evaluate_dp(chars)
            if dp.estimated_planning_time < remaining * 0.8:
                return dp, None

        if remaining > 2.0:
            lp = self._evaluate_lp(chars)
            if lp.estimated_planning_time < remaining * 0.8:
                return lp, None

        hybrid = RepairStrategy(
            planner_type=PlannerType.HYBRID,
            estimated_planning_time=elapsed,
            estimated_plan_quality=greedy.estimated_plan_quality,
            estimated_execution_time=greedy.estimated_execution_time,
            parameters={"initial_strategy": "greedy", "budget_used": elapsed},
            rationale="Adaptive: greedy initial, no time for improvement",
            confidence=greedy.confidence,
        )
        return hybrid, greedy_plan

    # ── Strategy Evaluation ───────────────────────────────────────

    def _evaluate_dp(self, chars: PipelineCharacteristics) -> RepairStrategy:
        """Evaluate the DP strategy for the given characteristics."""
        if not chars.is_dag:
            return RepairStrategy(
                planner_type=PlannerType.DP,
                estimated_planning_time=float("inf"),
                estimated_plan_quality=0.0,
                rationale="DP requires acyclic graph",
                confidence=0.0,
            )

        k = min(chars.affected_node_count, 20)
        planning_time = chars.node_count * (2 ** k) * 1e-6

        return RepairStrategy(
            planner_type=PlannerType.DP,
            estimated_planning_time=planning_time,
            estimated_plan_quality=1.0,
            estimated_execution_time=chars.estimated_total_cost * 0.5,
            parameters={
                "max_columns_tracked": k,
                "enable_annihilation": True,
            },
            rationale=f"DP optimal planner: O(|V|·2^k) with k={k}",
            confidence=0.95 if planning_time < 60 else 0.5,
        )

    def _evaluate_lp(self, chars: PipelineCharacteristics) -> RepairStrategy:
        """Evaluate the LP strategy."""
        planning_time = (
            chars.node_count * chars.edge_count * 1e-5
            + chars.affected_node_count * 1e-4
        )

        quality = 1.0 - (
            0.1 * (1 + (chars.affected_node_count / max(chars.node_count, 1)))
        )
        quality = max(0.5, min(1.0, quality))

        return RepairStrategy(
            planner_type=PlannerType.LP,
            estimated_planning_time=planning_time,
            estimated_plan_quality=quality,
            estimated_execution_time=chars.estimated_total_cost * 0.6,
            parameters={
                "rounding_threshold": 0.5,
                "max_local_search": 100,
            },
            rationale=f"LP relaxation: (ln k + 1)-approximation for {chars.node_count} nodes",
            confidence=0.8,
        )

    def _evaluate_greedy(
        self,
        chars: PipelineCharacteristics,
    ) -> RepairStrategy:
        """Evaluate the greedy strategy."""
        planning_time = chars.affected_node_count * 1e-5

        quality = 0.6 + 0.2 * (1.0 / (1 + chars.fan_out_max * 0.1))

        return RepairStrategy(
            planner_type=PlannerType.GREEDY,
            estimated_planning_time=planning_time,
            estimated_plan_quality=quality,
            estimated_execution_time=chars.estimated_total_cost * 0.8,
            parameters={
                "annihilation_enabled": True,
                "post_process": True,
            },
            rationale=f"Greedy: fast O(n log n) for {chars.affected_node_count} affected nodes",
            confidence=0.7,
        )

    def _evaluate_hybrid(
        self,
        chars: PipelineCharacteristics,
    ) -> RepairStrategy:
        """Evaluate the hybrid strategy."""
        greedy = self._evaluate_greedy(chars)
        dp = self._evaluate_dp(chars)
        lp = self._evaluate_lp(chars)

        if dp.confidence > 0 and dp.estimated_planning_time < 10:
            best_exact = dp
        else:
            best_exact = lp

        planning_time = greedy.estimated_planning_time + best_exact.estimated_planning_time * 0.5
        quality = greedy.estimated_plan_quality * 0.3 + best_exact.estimated_plan_quality * 0.7

        return RepairStrategy(
            planner_type=PlannerType.HYBRID,
            estimated_planning_time=planning_time,
            estimated_plan_quality=quality,
            estimated_execution_time=chars.estimated_total_cost * 0.55,
            parameters={
                "initial": "greedy",
                "improve_with": best_exact.planner_type.value,
            },
            rationale=f"Hybrid: greedy start, improve with {best_exact.planner_type.value}",
            confidence=0.85,
        )

    # ── Strategy Selection Helpers ────────────────────────────────

    def _select_for_cyclic(
        self,
        chars: PipelineCharacteristics,
        time_budget: float,
    ) -> RepairStrategy:
        """Select strategy for cyclic graphs."""
        lp = self._evaluate_lp(chars)
        if lp.estimated_planning_time < time_budget * 0.8:
            return lp

        greedy = self._evaluate_greedy(chars)
        greedy.rationale = "Greedy fallback: LP too slow for cyclic graph"
        return greedy

    def _select_for_small(
        self,
        chars: PipelineCharacteristics,
        time_budget: float,
    ) -> RepairStrategy:
        """Select strategy for small DAGs."""
        dp = self._evaluate_dp(chars)
        if dp.estimated_planning_time < time_budget * 0.8:
            return dp

        greedy = self._evaluate_greedy(chars)
        greedy.rationale = "Greedy: DP too slow even for small graph"
        return greedy

    def _select_for_medium(
        self,
        chars: PipelineCharacteristics,
        time_budget: float,
    ) -> RepairStrategy:
        """Select strategy for medium-sized pipelines."""
        dp = self._evaluate_dp(chars)
        if dp.estimated_planning_time < time_budget * 0.5:
            return dp

        hybrid = self._evaluate_hybrid(chars)
        if hybrid.estimated_planning_time < time_budget * 0.8:
            return hybrid

        return self._evaluate_greedy(chars)

    def _select_for_large(
        self,
        chars: PipelineCharacteristics,
        time_budget: float,
    ) -> RepairStrategy:
        """Select strategy for large pipelines."""
        greedy = self._evaluate_greedy(chars)

        if time_budget > 60:
            hybrid = self._evaluate_hybrid(chars)
            if hybrid.estimated_planning_time < time_budget * 0.8:
                return hybrid

        return greedy

    def _rank_strategies(
        self,
        strategies: Dict[str, RepairStrategy],
    ) -> str:
        """Rank strategies and return the name of the best one."""
        scores: Dict[str, float] = {}

        for name, strategy in strategies.items():
            if strategy.confidence == 0:
                scores[name] = -1.0
                continue

            score = (
                strategy.estimated_plan_quality * 40.0
                + strategy.confidence * 30.0
                - strategy.estimated_planning_time * 2.0
                - strategy.estimated_execution_time * 1.0
            )
            scores[name] = score

        return max(scores, key=lambda k: scores[k])


# =====================================================================
# Convenience Functions
# =====================================================================


def select_strategy(
    graph: Any,
    perturbation: Any = None,
    time_budget: float = 30.0,
) -> RepairStrategy:
    """Convenience: select the best repair strategy."""
    selector = RepairStrategySelector(default_time_budget=time_budget)
    return selector.select_strategy(graph, perturbation)


def compare_strategies(
    graph: Any,
    perturbation: Any = None,
) -> StrategyComparison:
    """Convenience: compare all repair strategies."""
    selector = RepairStrategySelector()
    return selector.compare_strategies(graph, perturbation)
