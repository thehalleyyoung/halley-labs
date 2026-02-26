"""
Schedule optimization utilities for MARACE.

Provides local search, evolutionary search, gradient-based estimation,
schedule interpolation, and continuous timing optimization to find
worst-case adversarial schedules that minimize safety margins in
multi-agent interaction-race verification.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# ScheduleCandidate
# ======================================================================

@dataclass
class ScheduleCandidate:
    """A candidate schedule with evaluated fitness (lower = more adversarial)."""
    schedule: List[Dict[str, object]]
    safety_margin: float = 0.0
    fitness: float = float("inf")

# ======================================================================
# LocalScheduleSearch
# ======================================================================

class LocalScheduleSearch:
    """Hill-climbing in the schedule neighbourhood to find worst-case interleavings."""

    def __init__(
        self,
        safety_evaluator: Callable[[list, np.ndarray], float],
        neighborhood_size: int = 10,
        max_iterations: int = 100,
        timing_perturbation: float = 0.1,
    ) -> None:
        self._evaluator = safety_evaluator
        self._neighborhood_size = neighborhood_size
        self._max_iterations = max_iterations
        self._timing_perturbation = timing_perturbation
        self._rng = np.random.default_rng()

    def search(self, initial_schedule: list, initial_state: np.ndarray) -> ScheduleCandidate:
        """Run hill-climbing from *initial_schedule* and return the best candidate."""
        best_fitness = self._evaluate(initial_schedule, initial_state)
        best_schedule = copy.deepcopy(initial_schedule)
        for iteration in range(self._max_iterations):
            improved = False
            for _ in range(self._neighborhood_size):
                neighbour = self._generate_neighbor(best_schedule)
                fitness = self._evaluate(neighbour, initial_state)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_schedule = neighbour
                    improved = True
            if not improved:
                logger.debug("LocalScheduleSearch converged at iteration %d (fitness=%.6f)", iteration, best_fitness)
                break
        return ScheduleCandidate(schedule=best_schedule, safety_margin=best_fitness, fitness=best_fitness)

    def _generate_neighbor(self, schedule: list) -> list:
        """Swap two adjacent concurrent actions or perturb timing."""
        neighbour = copy.deepcopy(schedule)
        if len(neighbour) < 2:
            return neighbour
        if self._rng.random() < 0.5:
            idx = int(self._rng.integers(0, len(neighbour) - 1))
            neighbour[idx], neighbour[idx + 1] = neighbour[idx + 1], neighbour[idx]
        else:
            idx = int(self._rng.integers(0, len(neighbour)))
            current = float(neighbour[idx].get("timing_offset", 0.0))
            neighbour[idx] = dict(neighbour[idx])
            neighbour[idx]["timing_offset"] = current + float(self._rng.normal(0.0, self._timing_perturbation))
        return neighbour

    def _evaluate(self, schedule: list, state: np.ndarray) -> float:
        """Evaluate the safety margin for *schedule* starting from *state*."""
        return float(self._evaluator(schedule, state))

# ======================================================================
# GeneticScheduleSearch
# ======================================================================

class GeneticScheduleSearch:
    """Evolutionary search over schedules using tournament selection and order crossover."""

    def __init__(
        self,
        safety_evaluator: Callable[[list, np.ndarray], float],
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self._evaluator = safety_evaluator
        self._population_size = population_size
        self._generations = generations
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._elite_fraction = elite_fraction
        self._rng = np.random.default_rng(seed)

    def search(self, initial_schedules: List[list], initial_state: np.ndarray) -> ScheduleCandidate:
        """Evolve a population seeded from *initial_schedules* and return the best."""
        population = self._initialize_population(initial_schedules)
        population = self._evaluate_population(population, initial_state)
        for gen in range(self._generations):
            population.sort(key=lambda c: c.fitness)
            elite_count = max(1, int(self._elite_fraction * self._population_size))
            next_gen: List[ScheduleCandidate] = list(population[:elite_count])
            while len(next_gen) < self._population_size:
                p1, p2 = self._select_parents(population)
                child = self._crossover(p1, p2) if self._rng.random() < self._crossover_rate else copy.deepcopy(p1.schedule)
                if self._rng.random() < self._mutation_rate:
                    child = self._mutate(child)
                next_gen.append(ScheduleCandidate(schedule=child))
            population = self._evaluate_population(next_gen, initial_state)
            best = min(population, key=lambda c: c.fitness)
            logger.debug("Generation %d: best fitness=%.6f", gen, best.fitness)
        population.sort(key=lambda c: c.fitness)
        return population[0]

    def _initialize_population(self, seeds: List[list]) -> List[ScheduleCandidate]:
        """Build initial population, padding with mutated seeds if needed."""
        population: List[ScheduleCandidate] = [ScheduleCandidate(schedule=copy.deepcopy(s)) for s in seeds]
        while len(population) < self._population_size:
            base = seeds[int(self._rng.integers(0, len(seeds)))]
            population.append(ScheduleCandidate(schedule=self._mutate(copy.deepcopy(base))))
        return population[: self._population_size]

    def _select_parents(self, population: List[ScheduleCandidate]) -> Tuple[ScheduleCandidate, ScheduleCandidate]:
        """Tournament selection of two parents."""
        k = min(3, len(population))
        def _tournament() -> ScheduleCandidate:
            idx = self._rng.choice(len(population), size=k, replace=False)
            contenders = sorted([population[int(i)] for i in idx], key=lambda c: c.fitness)
            return contenders[0]
        return _tournament(), _tournament()

    def _crossover(self, parent1: ScheduleCandidate, parent2: ScheduleCandidate) -> list:
        """Order crossover preserving HB validity."""
        s1, s2 = parent1.schedule, parent2.schedule
        n = min(len(s1), len(s2))
        if n <= 1:
            return copy.deepcopy(s1 if s1 else s2)
        cx1 = int(self._rng.integers(0, n))
        cx2 = int(self._rng.integers(cx1 + 1, n + 1))
        child: List[Optional[Dict[str, object]]] = [None] * n
        child[cx1:cx2] = copy.deepcopy(s1[cx1:cx2])
        fill_idx = cx2 % n
        for entry in s2:
            if fill_idx == cx1:
                break
            if child[fill_idx] is None:
                child[fill_idx] = copy.deepcopy(entry)
                fill_idx = (fill_idx + 1) % n
        s2_iter = iter(s2)
        for i in range(n):
            if child[i] is None:
                entry = next(s2_iter, None)
                child[i] = copy.deepcopy(entry) if entry is not None else copy.deepcopy(s1[i])
        return [c for c in child if c is not None]

    def _mutate(self, schedule: list) -> list:
        """Random swap of two actions or timing perturbation."""
        mutated = copy.deepcopy(schedule)
        if len(mutated) < 2:
            return mutated
        if self._rng.random() < 0.5:
            i, j = self._rng.choice(len(mutated), size=2, replace=False)
            mutated[int(i)], mutated[int(j)] = mutated[int(j)], mutated[int(i)]
        else:
            idx = int(self._rng.integers(0, len(mutated)))
            current = float(mutated[idx].get("timing_offset", 0.0))
            mutated[idx] = dict(mutated[idx])
            mutated[idx]["timing_offset"] = current + float(self._rng.normal(0.0, 0.1))
        return mutated

    def _evaluate_population(self, population: List[ScheduleCandidate], state: np.ndarray) -> List[ScheduleCandidate]:
        """Evaluate fitness for every candidate in *population*."""
        for candidate in population:
            margin = float(self._evaluator(candidate.schedule, state))
            candidate.safety_margin = margin
            candidate.fitness = margin
        return population

# ======================================================================
# GradientScheduleEstimate
# ======================================================================

class GradientScheduleEstimate:
    """Estimate gradient of safety margin w.r.t. timing parameters via finite differences."""

    def __init__(
        self,
        safety_evaluator: Callable[[list, np.ndarray], float],
        epsilon: float = 1e-4,
    ) -> None:
        self._evaluator = safety_evaluator
        self._epsilon = epsilon

    def estimate_gradient(self, schedule: list, state: np.ndarray) -> np.ndarray:
        """Central finite-difference gradient, one component per timing_offset."""
        n = len(schedule)
        gradient = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sched_plus = copy.deepcopy(schedule)
            sched_minus = copy.deepcopy(schedule)
            t_i = float(sched_plus[i].get("timing_offset", 0.0))
            sched_plus[i] = dict(sched_plus[i])
            sched_plus[i]["timing_offset"] = t_i + self._epsilon
            sched_minus[i] = dict(sched_minus[i])
            sched_minus[i]["timing_offset"] = t_i - self._epsilon
            margin_plus = float(self._evaluator(sched_plus, state))
            margin_minus = float(self._evaluator(sched_minus, state))
            gradient[i] = (margin_plus - margin_minus) / (2.0 * self._epsilon)
        return gradient

    def descent_step(self, schedule: list, state: np.ndarray, step_size: float = 0.01) -> list:
        """Take one gradient-descent step to reduce the safety margin."""
        gradient = self.estimate_gradient(schedule, state)
        updated = copy.deepcopy(schedule)
        for i in range(len(updated)):
            current = float(updated[i].get("timing_offset", 0.0))
            updated[i] = dict(updated[i])
            updated[i]["timing_offset"] = current - step_size * gradient[i]
        return updated

# ======================================================================
# ScheduleInterpolation
# ======================================================================

class ScheduleInterpolation:
    """Binary search on timing-parameter interpolation to find the safe/unsafe boundary."""

    def __init__(
        self,
        safety_evaluator: Callable[[list, np.ndarray], float],
        tolerance: float = 1e-4,
        max_bisections: int = 50,
    ) -> None:
        self._evaluator = safety_evaluator
        self._tolerance = tolerance
        self._max_bisections = max_bisections

    def interpolate(
        self, safe_schedule: list, unsafe_schedule: list, state: np.ndarray,
    ) -> Tuple[list, float]:
        """Return ``(boundary_schedule, boundary_margin)`` at the safe/unsafe boundary."""
        alpha_lo, alpha_hi = 0.0, 1.0
        for iteration in range(self._max_bisections):
            alpha_mid = (alpha_lo + alpha_hi) / 2.0
            mid_sched = self._lerp_schedules(safe_schedule, unsafe_schedule, alpha_mid)
            margin = float(self._evaluator(mid_sched, state))
            if margin >= 0.0:
                alpha_lo = alpha_mid
            else:
                alpha_hi = alpha_mid
            if (alpha_hi - alpha_lo) < self._tolerance:
                logger.debug("ScheduleInterpolation converged at iteration %d (alpha=%.6f)", iteration, alpha_mid)
                break
        alpha_boundary = (alpha_lo + alpha_hi) / 2.0
        boundary_sched = self._lerp_schedules(safe_schedule, unsafe_schedule, alpha_boundary)
        boundary_margin = float(self._evaluator(boundary_sched, state))
        return boundary_sched, boundary_margin

    @staticmethod
    def _lerp_schedules(s1: list, s2: list, alpha: float) -> list:
        """Linearly interpolate timing offsets: ``(1-α)*t1 + α*t2``."""
        n = min(len(s1), len(s2))
        result: list = []
        for i in range(n):
            t1 = float(s1[i].get("timing_offset", 0.0))
            t2 = float(s2[i].get("timing_offset", 0.0))
            entry = dict(s1[i])
            entry["timing_offset"] = (1.0 - alpha) * t1 + alpha * t2
            result.append(entry)
        for i in range(n, len(s1)):
            result.append(dict(s1[i]))
        for i in range(n, len(s2)):
            result.append(dict(s2[i]))
        return result

# ======================================================================
# TimingOptimizer
# ======================================================================

class TimingOptimizer:
    """Gradient descent on continuous timing offsets for worst-case interleaving.

    Uses :class:`GradientScheduleEstimate` internally for derivative estimation.
    """

    def __init__(
        self,
        safety_evaluator: Callable[[list, np.ndarray], float],
        max_iterations: int = 200,
        learning_rate: float = 0.01,
    ) -> None:
        self._evaluator = safety_evaluator
        self._max_iterations = max_iterations
        self._learning_rate = learning_rate
        self._grad_estimator = GradientScheduleEstimate(safety_evaluator)

    def optimize(self, schedule: list, state: np.ndarray) -> ScheduleCandidate:
        """Run gradient descent on timing offsets and return the best candidate."""
        best_schedule = copy.deepcopy(schedule)
        best_margin = float(self._evaluator(best_schedule, state))
        current_schedule = copy.deepcopy(schedule)
        for iteration in range(self._max_iterations):
            gradient = self._grad_estimator.estimate_gradient(current_schedule, state)
            grad_norm = float(np.linalg.norm(gradient))
            if grad_norm < 1e-10:
                logger.debug("TimingOptimizer converged at iteration %d (grad_norm=%.2e)", iteration, grad_norm)
                break
            updated = copy.deepcopy(current_schedule)
            for i in range(len(updated)):
                current_t = float(updated[i].get("timing_offset", 0.0))
                updated[i] = dict(updated[i])
                updated[i]["timing_offset"] = current_t - self._learning_rate * gradient[i]
            margin = float(self._evaluator(updated, state))
            if margin < best_margin:
                best_margin = margin
                best_schedule = copy.deepcopy(updated)
            current_schedule = updated
        return ScheduleCandidate(schedule=best_schedule, safety_margin=best_margin, fitness=best_margin)
