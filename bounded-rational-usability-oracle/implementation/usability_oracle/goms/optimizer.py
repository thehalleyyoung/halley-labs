"""
usability_oracle.goms.optimizer — GOMS-based UI optimization.

Identifies redundant operators, suggests method improvements, computes
the Pareto frontier of time/error/learning tradeoffs, and performs
multi-objective optimization using an NSGA-II-like approach with
constraint satisfaction for accessibility requirements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from usability_oracle.core.types import BoundingBox
from usability_oracle.core.constants import MINIMUM_TARGET_SIZE_PX
from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    GomsTrace,
    OperatorType,
)
from usability_oracle.goms.analyzer import (
    AnalyzerConfig,
    compute_critical_path_time,
    estimate_error_probability,
    estimate_learning_time,
    predict_execution_time,
)


# ═══════════════════════════════════════════════════════════════════════════
# Optimization suggestion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OptimizationSuggestion:
    """A single optimization suggestion."""

    description: str
    expected_time_saving_s: float
    operator_affected: str
    confidence: float
    """Confidence in the suggestion (0-1)."""
    category: str
    """One of: redundancy, target_size, method_simplification, layout."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "expected_time_saving_s": self.expected_time_saving_s,
            "operator_affected": self.operator_affected,
            "confidence": self.confidence,
            "category": self.category,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Pareto solution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParetoSolution:
    """A point on the Pareto frontier."""

    methods: Tuple[GomsMethod, ...]
    execution_time_s: float
    error_probability: float
    learning_time_s: float

    @property
    def objectives(self) -> Tuple[float, float, float]:
        return (self.execution_time_s, self.error_probability, self.learning_time_s)


# ═══════════════════════════════════════════════════════════════════════════
# Redundant operator identification
# ═══════════════════════════════════════════════════════════════════════════

def find_redundant_operators(
    trace: GomsTrace,
) -> List[Tuple[int, int, GomsOperator, str]]:
    """Identify redundant operators in a trace's method sequences.

    Returns a list of (method_index, operator_index, operator, reason).
    """
    redundancies: List[Tuple[int, int, GomsOperator, str]] = []

    for mi, method in enumerate(trace.methods_selected):
        ops = method.operators
        for oi, op in enumerate(ops):
            # Consecutive homing with no intervening device use
            if op.op_type == OperatorType.H and oi > 0:
                prev = ops[oi - 1]
                if prev.op_type == OperatorType.H:
                    redundancies.append((
                        mi, oi, op, "Consecutive homing — only one needed",
                    ))

            # Button release immediately followed by button press on same target
            if (
                op.op_type == OperatorType.B
                and op.parameters.get("action") == "release"
                and oi + 1 < len(ops)
                and ops[oi + 1].op_type == OperatorType.B
                and ops[oi + 1].target_id == op.target_id
            ):
                redundancies.append((
                    mi, oi, op,
                    f"Redundant release+press on {op.target_id}",
                ))

            # M operator with duration 0 (sometimes artifact of construction)
            if op.op_type == OperatorType.M and op.duration_s <= 0.0:
                redundancies.append((
                    mi, oi, op, "Zero-duration mental operator",
                ))

            # Pointing to a target already under the cursor (distance ≈ 0)
            if (
                op.op_type == OperatorType.P
                and op.parameters.get("distance_px", 999) < 2.0
            ):
                redundancies.append((
                    mi, oi, op,
                    "Pointing to co-located target (distance < 2px)",
                ))

    return redundancies


# ═══════════════════════════════════════════════════════════════════════════
# Method improvement suggestions
# ═══════════════════════════════════════════════════════════════════════════

def suggest_improvements(
    model: GomsModel,
    trace: GomsTrace,
    *,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> List[OptimizationSuggestion]:
    """Suggest improvements to the UI that reduce task time.

    Analyses bottleneck operators and proposes concrete changes.
    """
    suggestions: List[OptimizationSuggestion] = []

    # 1. Find bottleneck operators (> 20% of total time)
    total_time = trace.total_time_s
    if total_time <= 0:
        return suggestions

    for method in trace.methods_selected:
        for op in method.operators:
            frac = op.duration_s / total_time
            if frac > 0.20:
                if op.op_type == OperatorType.P:
                    # Pointing bottleneck → increase target size or reduce distance
                    target_w = op.parameters.get("target_width_px", 0)
                    if target_w < MINIMUM_TARGET_SIZE_PX:
                        saving = op.duration_s * 0.3
                        suggestions.append(OptimizationSuggestion(
                            description=(
                                f"Increase target size of {op.target_id} "
                                f"from {target_w:.0f}px to {MINIMUM_TARGET_SIZE_PX}px"
                            ),
                            expected_time_saving_s=saving,
                            operator_affected=op.description,
                            confidence=0.85,
                            category="target_size",
                        ))
                    dist = op.parameters.get("distance_px", 0)
                    if dist > 200:
                        saving = op.duration_s * 0.25
                        suggestions.append(OptimizationSuggestion(
                            description=(
                                f"Move {op.target_id} closer to preceding "
                                f"target (distance={dist:.0f}px)"
                            ),
                            expected_time_saving_s=saving,
                            operator_affected=op.description,
                            confidence=0.70,
                            category="layout",
                        ))

                elif op.op_type == OperatorType.M:
                    n_choices = op.parameters.get("n_choices", 0)
                    if n_choices > 7:
                        saving = op.duration_s * 0.4
                        suggestions.append(OptimizationSuggestion(
                            description=(
                                f"Reduce number of choices from {n_choices} "
                                f"(group into categories or use search)"
                            ),
                            expected_time_saving_s=saving,
                            operator_affected=op.description,
                            confidence=0.80,
                            category="method_simplification",
                        ))

                elif op.op_type == OperatorType.R:
                    if op.duration_s > 1.0:
                        suggestions.append(OptimizationSuggestion(
                            description=(
                                f"Reduce system response latency "
                                f"({op.duration_s:.2f}s → <1s)"
                            ),
                            expected_time_saving_s=op.duration_s - 0.5,
                            operator_affected=op.description,
                            confidence=0.90,
                            category="method_simplification",
                        ))

    # 2. Identify redundancies
    for _mi, _oi, op, reason in find_redundant_operators(trace):
        suggestions.append(OptimizationSuggestion(
            description=f"Remove redundant operator: {reason}",
            expected_time_saving_s=op.duration_s,
            operator_affected=op.description,
            confidence=0.95,
            category="redundancy",
        ))

    # 3. Suggest keyboard shortcuts for repeated navigation
    nav_count = sum(
        1 for m in trace.methods_selected
        for op in m.operators
        if op.op_type == OperatorType.P
    )
    if nav_count > 5:
        avg_point_time = sum(
            op.duration_s for m in trace.methods_selected
            for op in m.operators if op.op_type == OperatorType.P
        ) / nav_count
        suggestions.append(OptimizationSuggestion(
            description=(
                f"Add keyboard shortcut to reduce {nav_count} pointing "
                f"operations (avg {avg_point_time:.2f}s each)"
            ),
            expected_time_saving_s=avg_point_time * nav_count * 0.5,
            operator_affected="Multiple pointing operators",
            confidence=0.65,
            category="method_simplification",
        ))

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════
# Pareto frontier computation
# ═══════════════════════════════════════════════════════════════════════════

def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """Return True if *a* Pareto-dominates *b* (all ≤ and at least one <)."""
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            at_least_one_better = True
    return at_least_one_better


def compute_pareto_frontier(
    model: GomsModel,
    *,
    config: AnalyzerConfig = AnalyzerConfig(),
    n_samples: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> List[ParetoSolution]:
    """Compute the Pareto frontier of time/error/learning tradeoffs.

    Enumerates method combinations (or samples if space is large) and
    extracts the non-dominated set.

    Parameters
    ----------
    model : GomsModel
        GOMS model with potentially multiple methods per goal.
    config : AnalyzerConfig
        Analyzer configuration.
    n_samples : int
        Maximum number of combinations to evaluate.
    rng : Optional[np.random.Generator]
        Random number generator.

    Returns
    -------
    list[ParetoSolution]
        Non-dominated solutions.
    """
    gen = rng or np.random.default_rng(42)

    # Collect goals with multiple methods
    goal_methods: Dict[str, List[GomsMethod]] = {}
    for goal in model.goals:
        if goal.is_leaf:
            methods = list(model.methods_for_goal(goal.goal_id))
            if methods:
                goal_methods[goal.goal_id] = methods

    if not goal_methods:
        return []

    # Enumerate or sample combinations
    goal_ids = sorted(goal_methods.keys())
    n_combos = 1
    for gid in goal_ids:
        n_combos *= len(goal_methods[gid])

    solutions: List[ParetoSolution] = []

    if n_combos <= n_samples:
        # Full enumeration
        import itertools
        ranges = [range(len(goal_methods[gid])) for gid in goal_ids]
        for combo in itertools.product(*ranges):
            selected = tuple(
                goal_methods[goal_ids[i]][combo[i]]
                for i in range(len(goal_ids))
            )
            solutions.append(_evaluate_combo(selected, model, config))
    else:
        # Random sampling
        for _ in range(n_samples):
            selected = tuple(
                goal_methods[gid][gen.integers(len(goal_methods[gid]))]
                for gid in goal_ids
            )
            solutions.append(_evaluate_combo(selected, model, config))

    # Extract Pareto front
    return _extract_pareto_front(solutions)


def _evaluate_combo(
    methods: Tuple[GomsMethod, ...],
    model: GomsModel,
    config: AnalyzerConfig,
) -> ParetoSolution:
    """Evaluate a method combination on all three objectives."""
    exec_time = predict_execution_time(methods, config=config)
    err_prob = estimate_error_probability(methods, config=config)
    learn_time = estimate_learning_time(model, config=config)
    return ParetoSolution(
        methods=methods,
        execution_time_s=exec_time,
        error_probability=err_prob,
        learning_time_s=learn_time,
    )


def _extract_pareto_front(
    solutions: Sequence[ParetoSolution],
) -> List[ParetoSolution]:
    """Extract the non-dominated front from a set of solutions."""
    front: List[ParetoSolution] = []
    for candidate in solutions:
        dominated = False
        for other in solutions:
            if other is candidate:
                continue
            if _dominates(other.objectives, candidate.objectives):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    # Deduplicate by objectives
    seen: Set[Tuple[float, float, float]] = set()
    unique: List[ParetoSolution] = []
    for sol in front:
        key = (round(sol.execution_time_s, 6),
               round(sol.error_probability, 6),
               round(sol.learning_time_s, 6))
        if key not in seen:
            seen.add(key)
            unique.append(sol)
    return unique


# ═══════════════════════════════════════════════════════════════════════════
# NSGA-II-like multi-objective optimization
# ═══════════════════════════════════════════════════════════════════════════

def nsga2_optimize(
    model: GomsModel,
    *,
    config: AnalyzerConfig = AnalyzerConfig(),
    population_size: int = 50,
    n_generations: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> List[ParetoSolution]:
    """Multi-objective optimization using NSGA-II-like approach.

    Evolves a population of method-selection vectors to find the
    Pareto-optimal set of time/error/learning tradeoffs.
    """
    gen = rng or np.random.default_rng(42)

    # Setup: goals with methods
    goal_methods: Dict[str, List[GomsMethod]] = {}
    for goal in model.goals:
        if goal.is_leaf:
            methods = list(model.methods_for_goal(goal.goal_id))
            if methods:
                goal_methods[goal.goal_id] = methods

    if not goal_methods:
        return []

    goal_ids = sorted(goal_methods.keys())
    n_goals = len(goal_ids)
    method_counts = [len(goal_methods[gid]) for gid in goal_ids]

    # Initialize population: each individual is a vector of method indices
    population = np.column_stack([
        gen.integers(0, mc, size=population_size)
        for mc in method_counts
    ])

    def _decode(individual: np.ndarray) -> Tuple[GomsMethod, ...]:
        return tuple(
            goal_methods[goal_ids[i]][int(individual[i])]
            for i in range(n_goals)
        )

    def _evaluate(individual: np.ndarray) -> Tuple[float, float, float]:
        methods = _decode(individual)
        et = predict_execution_time(methods, config=config)
        ep = estimate_error_probability(methods, config=config)
        lt = estimate_learning_time(model, config=config)
        return (et, ep, lt)

    for _generation in range(n_generations):
        # Evaluate
        fitnesses = np.array([_evaluate(ind) for ind in population])

        # Non-dominated sorting (simplified: just rank by front)
        fronts = _fast_non_dominated_sort(fitnesses)

        # Crowding distance
        crowding = np.zeros(len(population))
        for front in fronts:
            if len(front) < 3:
                for idx in front:
                    crowding[idx] = float("inf")
                continue
            for obj_idx in range(3):
                sorted_indices = sorted(front, key=lambda x: fitnesses[x, obj_idx])
                crowding[sorted_indices[0]] = float("inf")
                crowding[sorted_indices[-1]] = float("inf")
                obj_range = (
                    fitnesses[sorted_indices[-1], obj_idx]
                    - fitnesses[sorted_indices[0], obj_idx]
                )
                if obj_range > 0:
                    for k in range(1, len(sorted_indices) - 1):
                        crowding[sorted_indices[k]] += (
                            fitnesses[sorted_indices[k + 1], obj_idx]
                            - fitnesses[sorted_indices[k - 1], obj_idx]
                        ) / obj_range

        # Assign rank (front index)
        rank = np.zeros(len(population), dtype=int)
        for fi, front in enumerate(fronts):
            for idx in front:
                rank[idx] = fi

        # Tournament selection + crossover + mutation
        offspring = np.empty_like(population)
        for i in range(0, population_size, 2):
            p1 = _tournament_select(rank, crowding, gen)
            p2 = _tournament_select(rank, crowding, gen)
            c1, c2 = _crossover(population[p1], population[p2], gen)
            offspring[i] = _mutate(c1, method_counts, gen)
            if i + 1 < population_size:
                offspring[i + 1] = _mutate(c2, method_counts, gen)

        # Combine parent + offspring, select best
        combined = np.vstack([population, offspring])
        combined_fit = np.array([_evaluate(ind) for ind in combined])
        combined_fronts = _fast_non_dominated_sort(combined_fit)

        # Select top population_size
        new_pop: List[np.ndarray] = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= population_size:
                new_pop.extend(combined[idx] for idx in front)
            else:
                # Fill from this front by crowding distance
                cd = np.zeros(len(combined))
                for obj_idx in range(3):
                    sorted_idx = sorted(front, key=lambda x: combined_fit[x, obj_idx])
                    if len(sorted_idx) >= 2:
                        cd[sorted_idx[0]] = float("inf")
                        cd[sorted_idx[-1]] = float("inf")
                        obj_range = (
                            combined_fit[sorted_idx[-1], obj_idx]
                            - combined_fit[sorted_idx[0], obj_idx]
                        )
                        if obj_range > 0:
                            for k in range(1, len(sorted_idx) - 1):
                                cd[sorted_idx[k]] += (
                                    combined_fit[sorted_idx[k + 1], obj_idx]
                                    - combined_fit[sorted_idx[k - 1], obj_idx]
                                ) / obj_range
                remaining = population_size - len(new_pop)
                by_cd = sorted(front, key=lambda x: cd[x], reverse=True)
                new_pop.extend(combined[idx] for idx in by_cd[:remaining])
                break

        population = np.array(new_pop[:population_size])

    # Extract final Pareto front
    final_fitnesses = np.array([_evaluate(ind) for ind in population])
    fronts = _fast_non_dominated_sort(final_fitnesses)
    result: List[ParetoSolution] = []
    if fronts:
        for idx in fronts[0]:
            methods = _decode(population[idx])
            result.append(ParetoSolution(
                methods=methods,
                execution_time_s=final_fitnesses[idx, 0],
                error_probability=final_fitnesses[idx, 1],
                learning_time_s=final_fitnesses[idx, 2],
            ))
    return result


def _fast_non_dominated_sort(
    fitnesses: np.ndarray,
) -> List[List[int]]:
    """Fast non-dominated sorting (Deb et al. 2002)."""
    n = len(fitnesses)
    domination_count = [0] * n
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _np_dominates(fitnesses[i], fitnesses[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif _np_dominates(fitnesses[j], fitnesses[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    fi = 0
    while fronts[fi]:
        next_front: List[int] = []
        for i in fronts[fi]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        fi += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _np_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _tournament_select(
    rank: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Binary tournament selection by rank then crowding distance."""
    i, j = rng.integers(0, len(rank), size=2)
    if rank[i] < rank[j]:
        return int(i)
    if rank[j] < rank[i]:
        return int(j)
    return int(i) if crowding[i] >= crowding[j] else int(j)


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover."""
    mask = rng.random(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def _mutate(
    individual: np.ndarray,
    method_counts: List[int],
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """Mutate an individual by randomly changing method selections."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if rng.random() < mutation_rate and method_counts[i] > 1:
            mutated[i] = rng.integers(0, method_counts[i])
    return mutated


# ═══════════════════════════════════════════════════════════════════════════
# Accessibility constraint satisfaction
# ═══════════════════════════════════════════════════════════════════════════

def check_accessibility_constraints(
    trace: GomsTrace,
) -> List[Dict[str, Any]]:
    """Check accessibility constraints on a trace.

    Validates:
    - Target sizes meet WCAG minimums
    - Task time doesn't exceed reasonable limits
    - Error probability is below threshold

    Returns
    -------
    list[dict]
        Violated constraints.
    """
    violations: List[Dict[str, Any]] = []

    for method in trace.methods_selected:
        for op in method.operators:
            if op.op_type == OperatorType.P and op.target_bounds is not None:
                min_dim = min(op.target_bounds.width, op.target_bounds.height)
                if min_dim < MINIMUM_TARGET_SIZE_PX:
                    violations.append({
                        "constraint": "WCAG 2.5.8 Target Size",
                        "target_id": op.target_id,
                        "actual_size_px": min_dim,
                        "required_size_px": MINIMUM_TARGET_SIZE_PX,
                        "severity": "high",
                    })

    err_prob = trace.metadata.get("error_probability", 0.0)
    if err_prob > 0.30:
        violations.append({
            "constraint": "Error Rate Threshold",
            "actual": err_prob,
            "threshold": 0.30,
            "severity": "high",
        })

    if trace.total_time_s > 120.0:
        violations.append({
            "constraint": "Task Time Limit",
            "actual_s": trace.total_time_s,
            "threshold_s": 120.0,
            "severity": "medium",
        })

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# GomsOptimizerImpl — concrete optimizer implementation
# ═══════════════════════════════════════════════════════════════════════════

class GomsOptimizerImpl:
    """Concrete implementation of the GomsOptimizer protocol."""

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self._config = config or AnalyzerConfig()

    def identify_bottleneck_operators(
        self,
        trace: GomsTrace,
        *,
        threshold_fraction: float = 0.2,
    ) -> Sequence[str]:
        """Return operator descriptions consuming > threshold of total time."""
        if trace.total_time_s <= 0:
            return []
        bottlenecks: List[str] = []
        for method in trace.methods_selected:
            for op in method.operators:
                if op.duration_s / trace.total_time_s > threshold_fraction:
                    bottlenecks.append(
                        f"{op.op_type.name}: {op.description} "
                        f"({op.duration_s:.2f}s, "
                        f"{op.duration_s / trace.total_time_s:.0%} of total)"
                    )
        return bottlenecks

    def suggest_improvements(
        self,
        model: GomsModel,
        trace: GomsTrace,
    ) -> Sequence[Mapping[str, Any]]:
        """Suggest improvements to reduce task time."""
        suggestions = suggest_improvements(model, trace, config=self._config)
        return [s.to_dict() for s in suggestions]

    def compute_pareto_frontier(
        self,
        model: GomsModel,
        *,
        n_samples: int = 100,
    ) -> List[ParetoSolution]:
        """Compute Pareto frontier for the model."""
        return compute_pareto_frontier(
            model, config=self._config, n_samples=n_samples,
        )


__all__ = [
    "GomsOptimizerImpl",
    "OptimizationSuggestion",
    "ParetoSolution",
    "check_accessibility_constraints",
    "compute_pareto_frontier",
    "find_redundant_operators",
    "nsga2_optimize",
    "suggest_improvements",
]
