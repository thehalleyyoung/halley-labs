"""Multi-objective optimization for quality-diversity tradeoffs.

Implements NSGA-II, MOEA/D, hypervolume indicator selection, and
scalarization methods for finding Pareto-optimal diverse subsets.

Mathematical foundations:
- Pareto dominance: x dominates y iff x_i >= y_i for all i, strict for some
- Hypervolume: volume dominated by the Pareto front
- Tchebycheff: min_x max_i w_i |f_i(x) - z_i^*|
- NSGA-II: non-dominated sorting + crowding distance
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .kernels import Kernel, RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    """A solution in multi-objective optimization."""
    indices: List[int]  # selected item indices
    objectives: np.ndarray  # objective values
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParetoFront:
    """The Pareto front of non-dominated solutions."""
    solutions: List[Solution]
    hypervolume: float = 0.0
    n_objectives: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class MOOResult:
    """Result from multi-objective optimization."""
    pareto_front: ParetoFront
    best_scalarized: Optional[Solution] = None
    knee_point: Optional[Solution] = None
    n_generations: int = 0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

class ObjectiveFunction(ABC):
    """Base class for objective functions."""

    @abstractmethod
    def evaluate(self, indices: List[int]) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def maximize(self) -> bool:
        return True


class QualityObjective(ObjectiveFunction):
    """Mean quality of selected items."""

    def __init__(self, quality_scores: np.ndarray):
        self.quality_scores = quality_scores

    def evaluate(self, indices: List[int]) -> float:
        if len(indices) == 0:
            return 0.0
        return float(np.mean(self.quality_scores[indices]))

    @property
    def name(self) -> str:
        return "quality"


class DiversityObjective(ObjectiveFunction):
    """Log-det diversity of selected items."""

    def __init__(self, embeddings: np.ndarray, kernel: Optional[Kernel] = None):
        self.embeddings = embeddings
        self.kernel = kernel or RBFKernel(bandwidth=1.0)

    def evaluate(self, indices: List[int]) -> float:
        if len(indices) < 2:
            return 0.0
        K = self.kernel.gram_matrix(self.embeddings[indices])
        return log_det_safe(K)

    @property
    def name(self) -> str:
        return "diversity"


class CoverageObjective(ObjectiveFunction):
    """Coverage of reference distribution."""

    def __init__(
        self,
        embeddings: np.ndarray,
        reference: np.ndarray,
        radius: float = 1.0,
    ):
        self.embeddings = embeddings
        self.reference = reference
        self.radius = radius

    def evaluate(self, indices: List[int]) -> float:
        if len(indices) == 0:
            return 0.0
        selected = self.embeddings[indices]
        covered = 0
        for ref in self.reference:
            dists = np.linalg.norm(selected - ref, axis=1)
            if np.min(dists) <= self.radius:
                covered += 1
        return covered / len(self.reference)

    @property
    def name(self) -> str:
        return "coverage"


class NoveltyObjective(ObjectiveFunction):
    """Mean pairwise distance (novelty)."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def evaluate(self, indices: List[int]) -> float:
        if len(indices) < 2:
            return 0.0
        selected = self.embeddings[indices]
        n = len(selected)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.linalg.norm(selected[i] - selected[j])
                count += 1
        return total / max(count, 1)

    @property
    def name(self) -> str:
        return "novelty"


# ---------------------------------------------------------------------------
# Pareto dominance utilities
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if a Pareto-dominates b (all objectives maximized)."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(solutions: List[Solution]) -> List[List[int]]:
    """Non-dominated sorting for NSGA-II.

    Returns list of fronts (each front is a list of indices).
    """
    n = len(solutions)
    domination_count = [0] * n
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(solutions[i].objectives, solutions[j].objectives):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(solutions[j].objectives, solutions[i].objectives):
                dominated_by[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            solutions[i].rank = 0
            fronts[0].append(i)

    front_idx = 0
    while fronts[front_idx]:
        next_front: List[int] = []
        for i in fronts[front_idx]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    solutions[j].rank = front_idx + 1
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(solutions: List[Solution], front: List[int]) -> None:
    """Compute crowding distance for solutions in a front."""
    if len(front) <= 2:
        for idx in front:
            solutions[idx].crowding_distance = float("inf")
        return

    n_obj = solutions[front[0]].objectives.shape[0]
    for idx in front:
        solutions[idx].crowding_distance = 0.0

    for m in range(n_obj):
        # Sort by objective m
        sorted_front = sorted(front, key=lambda i: solutions[i].objectives[m])
        obj_min = solutions[sorted_front[0]].objectives[m]
        obj_max = solutions[sorted_front[-1]].objectives[m]
        obj_range = obj_max - obj_min

        solutions[sorted_front[0]].crowding_distance = float("inf")
        solutions[sorted_front[-1]].crowding_distance = float("inf")

        if obj_range < 1e-12:
            continue

        for i in range(1, len(sorted_front) - 1):
            idx = sorted_front[i]
            prev_val = solutions[sorted_front[i - 1]].objectives[m]
            next_val = solutions[sorted_front[i + 1]].objectives[m]
            solutions[idx].crowding_distance += (next_val - prev_val) / obj_range


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------

class NSGA2:
    """NSGA-II for multi-objective diversity-quality optimization.

    Each individual is a subset of items. Mutation = swap an item.
    Crossover = combine subsets.
    """

    def __init__(
        self,
        n_items: int,
        k: int,
        objectives: List[ObjectiveFunction],
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.RandomState(seed)

    def _random_solution(self) -> Solution:
        """Generate a random solution."""
        indices = sorted(self.rng.choice(self.n_items, size=self.k, replace=False).tolist())
        obj_vals = np.array([obj.evaluate(indices) for obj in self.objectives])
        return Solution(indices=indices, objectives=obj_vals)

    def _evaluate(self, indices: List[int]) -> np.ndarray:
        """Evaluate all objectives for a solution."""
        return np.array([obj.evaluate(indices) for obj in self.objectives])

    def _crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """Uniform crossover of two subsets."""
        combined = list(set(parent1.indices) | set(parent2.indices))
        if len(combined) <= self.k:
            child_indices = combined
        else:
            child_indices = sorted(self.rng.choice(combined, size=self.k, replace=False).tolist())
        obj_vals = self._evaluate(child_indices)
        return Solution(indices=child_indices, objectives=obj_vals)

    def _mutate(self, solution: Solution) -> Solution:
        """Mutation: swap one item with a random new one."""
        indices = solution.indices.copy()
        if self.rng.random() < self.mutation_rate:
            pos = self.rng.randint(len(indices))
            available = [i for i in range(self.n_items) if i not in indices]
            if available:
                indices[pos] = self.rng.choice(available)
                indices = sorted(indices)
        obj_vals = self._evaluate(indices)
        return Solution(indices=indices, objectives=obj_vals)

    def _tournament_select(self, population: List[Solution]) -> Solution:
        """Binary tournament selection based on rank and crowding."""
        i, j = self.rng.choice(len(population), size=2, replace=False)
        a, b = population[i], population[j]
        if a.rank < b.rank:
            return a
        elif b.rank < a.rank:
            return b
        elif a.crowding_distance > b.crowding_distance:
            return a
        else:
            return b

    def run(self, n_generations: int = 100) -> MOOResult:
        """Run NSGA-II optimization."""
        # Initialize population
        population = [self._random_solution() for _ in range(self.pop_size)]

        for gen in range(n_generations):
            # Create offspring
            offspring: List[Solution] = []
            for _ in range(self.pop_size):
                if self.rng.random() < self.crossover_rate:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                else:
                    child = self._tournament_select(population)
                child = self._mutate(child)
                offspring.append(child)

            # Combine
            combined = population + offspring

            # Non-dominated sorting
            fronts = non_dominated_sort(combined)

            # Fill next generation
            new_population: List[Solution] = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    crowding_distance(combined, front)
                    new_population.extend([combined[i] for i in front])
                else:
                    crowding_distance(combined, front)
                    remaining = self.pop_size - len(new_population)
                    sorted_front = sorted(
                        front, key=lambda i: combined[i].crowding_distance, reverse=True
                    )
                    new_population.extend([combined[i] for i in sorted_front[:remaining]])
                    break

            population = new_population

        # Extract Pareto front
        fronts = non_dominated_sort(population)
        pareto_solutions = [population[i] for i in fronts[0]]

        hv = self._compute_hypervolume(pareto_solutions)
        knee = self._find_knee_point(pareto_solutions)

        pf = ParetoFront(
            solutions=pareto_solutions,
            hypervolume=hv,
            n_objectives=self.n_obj,
        )

        return MOOResult(
            pareto_front=pf,
            knee_point=knee,
            n_generations=n_generations,
            metadata={"algorithm": "NSGA-II", "pop_size": self.pop_size},
        )

    def _compute_hypervolume(
        self, solutions: List[Solution], ref_point: Optional[np.ndarray] = None,
    ) -> float:
        """Compute hypervolume indicator (2D or approximate higher-D)."""
        if len(solutions) == 0:
            return 0.0
        n_obj = solutions[0].objectives.shape[0]

        if ref_point is None:
            ref_point = np.zeros(n_obj)

        if n_obj == 2:
            return self._hypervolume_2d(solutions, ref_point)
        return self._hypervolume_monte_carlo(solutions, ref_point)

    def _hypervolume_2d(
        self, solutions: List[Solution], ref_point: np.ndarray,
    ) -> float:
        """Exact 2D hypervolume via sweep."""
        points = sorted(
            [s.objectives for s in solutions],
            key=lambda p: p[0],
            reverse=True,
        )
        hv = 0.0
        prev_y = ref_point[1]
        for p in points:
            if p[0] > ref_point[0] and p[1] > prev_y:
                hv += (p[0] - ref_point[0]) * (p[1] - prev_y)
                prev_y = p[1]
        return hv

    def _hypervolume_monte_carlo(
        self,
        solutions: List[Solution],
        ref_point: np.ndarray,
        n_samples: int = 10000,
    ) -> float:
        """Monte Carlo hypervolume estimation for higher dimensions."""
        obj_matrix = np.array([s.objectives for s in solutions])
        obj_max = np.max(obj_matrix, axis=0)
        n_obj = len(ref_point)

        # Sample uniformly in [ref, max] box
        volume = np.prod(obj_max - ref_point)
        if volume <= 0:
            return 0.0

        count = 0
        for _ in range(n_samples):
            point = ref_point + self.rng.random(n_obj) * (obj_max - ref_point)
            # Check if dominated by any solution
            for s in solutions:
                if np.all(s.objectives >= point):
                    count += 1
                    break

        return volume * count / n_samples

    def _find_knee_point(self, solutions: List[Solution]) -> Optional[Solution]:
        """Find knee point of Pareto front (maximum marginal rate change)."""
        if len(solutions) < 3:
            return solutions[0] if solutions else None

        n_obj = solutions[0].objectives.shape[0]
        if n_obj != 2:
            # For higher dimensions, use distance to utopia point
            utopia = np.max(np.array([s.objectives for s in solutions]), axis=0)
            best = None
            best_dist = float("inf")
            for s in solutions:
                dist = np.linalg.norm(s.objectives - utopia)
                if dist < best_dist:
                    best_dist = dist
                    best = s
            return best

        # 2D case: maximum curvature
        sorted_sols = sorted(solutions, key=lambda s: s.objectives[0])
        max_angle = 0.0
        knee = sorted_sols[0]

        for i in range(1, len(sorted_sols) - 1):
            prev_obj = sorted_sols[i - 1].objectives
            curr_obj = sorted_sols[i].objectives
            next_obj = sorted_sols[i + 1].objectives

            v1 = curr_obj - prev_obj
            v2 = next_obj - curr_obj

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            angle = math.acos(np.clip(cos_angle, -1, 1))

            if angle > max_angle:
                max_angle = angle
                knee = sorted_sols[i]

        return knee


# ---------------------------------------------------------------------------
# MOEA/D (Decomposition)
# ---------------------------------------------------------------------------

class MOEAD:
    """MOEA/D: Multi-Objective EA based on Decomposition.

    Decomposes multi-objective problem into scalar subproblems using
    weight vectors and optimizes them cooperatively.
    """

    def __init__(
        self,
        n_items: int,
        k: int,
        objectives: List[ObjectiveFunction],
        n_weight_vectors: int = 50,
        n_neighbors: int = 10,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n_vectors = n_weight_vectors
        self.n_neighbors = n_neighbors
        self.rng = np.random.RandomState(seed)

        # Generate weight vectors
        self.weight_vectors = self._generate_weights()
        # Compute neighborhoods
        self.neighborhoods = self._compute_neighborhoods()

    def _generate_weights(self) -> np.ndarray:
        """Generate uniformly distributed weight vectors."""
        if self.n_obj == 2:
            weights = np.zeros((self.n_vectors, 2))
            for i in range(self.n_vectors):
                w = i / (self.n_vectors - 1)
                weights[i] = [w, 1 - w]
            return weights
        else:
            # Random weights on simplex
            weights = self.rng.dirichlet(np.ones(self.n_obj), size=self.n_vectors)
            return weights

    def _compute_neighborhoods(self) -> List[List[int]]:
        """Compute neighborhoods based on weight vector distances."""
        dists = np.zeros((self.n_vectors, self.n_vectors))
        for i in range(self.n_vectors):
            for j in range(self.n_vectors):
                dists[i, j] = np.linalg.norm(
                    self.weight_vectors[i] - self.weight_vectors[j]
                )
        neighborhoods = []
        for i in range(self.n_vectors):
            neighbors = np.argsort(dists[i])[:self.n_neighbors]
            neighborhoods.append(neighbors.tolist())
        return neighborhoods

    def _evaluate(self, indices: List[int]) -> np.ndarray:
        return np.array([obj.evaluate(indices) for obj in self.objectives])

    def _tchebycheff(
        self, obj_vals: np.ndarray, weights: np.ndarray, z_ideal: np.ndarray,
    ) -> float:
        """Tchebycheff scalarization (to minimize)."""
        return float(np.max(weights * np.abs(obj_vals - z_ideal)))

    def _random_solution(self) -> Tuple[List[int], np.ndarray]:
        indices = sorted(self.rng.choice(self.n_items, size=self.k, replace=False).tolist())
        return indices, self._evaluate(indices)

    def _mutate(self, indices: List[int]) -> List[int]:
        new_indices = indices.copy()
        pos = self.rng.randint(len(new_indices))
        available = [i for i in range(self.n_items) if i not in new_indices]
        if available:
            new_indices[pos] = self.rng.choice(available)
        return sorted(new_indices)

    def run(self, n_generations: int = 100) -> MOOResult:
        """Run MOEA/D."""
        # Initialize
        population: List[Tuple[List[int], np.ndarray]] = [
            self._random_solution() for _ in range(self.n_vectors)
        ]
        z_ideal = np.max(np.array([obj for _, obj in population]), axis=0)

        for gen in range(n_generations):
            for i in range(self.n_vectors):
                # Select parents from neighborhood
                neighbors = self.neighborhoods[i]
                j, k_idx = self.rng.choice(neighbors, size=2, replace=False)

                # Crossover
                parent1 = population[j][0]
                parent2 = population[k_idx][0]
                combined = list(set(parent1) | set(parent2))
                if len(combined) > self.k:
                    child = sorted(self.rng.choice(combined, size=self.k, replace=False).tolist())
                else:
                    child = combined

                # Mutate
                child = self._mutate(child)
                child_obj = self._evaluate(child)

                # Update ideal point
                z_ideal = np.maximum(z_ideal, child_obj)

                # Update neighbors
                for nb in neighbors:
                    w = self.weight_vectors[nb]
                    curr_te = self._tchebycheff(population[nb][1], w, z_ideal)
                    child_te = self._tchebycheff(child_obj, w, z_ideal)
                    if child_te < curr_te:
                        population[nb] = (child, child_obj)

        # Extract Pareto front
        solutions = [
            Solution(indices=idx, objectives=obj)
            for idx, obj in population
        ]
        fronts = non_dominated_sort(solutions)
        pareto = [solutions[i] for i in fronts[0]]

        return MOOResult(
            pareto_front=ParetoFront(solutions=pareto, n_objectives=self.n_obj),
            n_generations=n_generations,
            metadata={"algorithm": "MOEA/D"},
        )


# ---------------------------------------------------------------------------
# Hypervolume Indicator Selection
# ---------------------------------------------------------------------------

class HypervolumeSelection:
    """Indicator-based selection using hypervolume contribution."""

    def __init__(
        self,
        n_items: int,
        k: int,
        objectives: List[ObjectiveFunction],
        population_size: int = 50,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.pop_size = population_size
        self.rng = np.random.RandomState(seed)

    def _evaluate(self, indices: List[int]) -> np.ndarray:
        return np.array([obj.evaluate(indices) for obj in self.objectives])

    def _hypervolume_contribution(
        self,
        solution: Solution,
        others: List[Solution],
        ref_point: np.ndarray,
    ) -> float:
        """Compute hypervolume contribution of a solution."""
        # HV with solution - HV without
        all_sols = others + [solution]
        hv_with = self._compute_hv(all_sols, ref_point)
        hv_without = self._compute_hv(others, ref_point)
        return hv_with - hv_without

    def _compute_hv(
        self, solutions: List[Solution], ref_point: np.ndarray,
    ) -> float:
        """Simple hypervolume computation."""
        if len(solutions) == 0:
            return 0.0
        if self.n_obj == 2:
            points = sorted([s.objectives for s in solutions], key=lambda p: p[0], reverse=True)
            hv = 0.0
            prev_y = ref_point[1]
            for p in points:
                if p[0] > ref_point[0] and p[1] > prev_y:
                    hv += (p[0] - ref_point[0]) * (p[1] - prev_y)
                    prev_y = p[1]
            return hv
        # MC approximation for higher D
        obj_matrix = np.array([s.objectives for s in solutions])
        obj_max = np.max(obj_matrix, axis=0)
        volume = float(np.prod(np.maximum(obj_max - ref_point, 0)))
        if volume <= 0:
            return 0.0
        count = 0
        n_samples = 1000
        for _ in range(n_samples):
            point = ref_point + self.rng.random(self.n_obj) * (obj_max - ref_point)
            for s in solutions:
                if np.all(s.objectives >= point):
                    count += 1
                    break
        return volume * count / n_samples

    def run(self, n_generations: int = 100) -> MOOResult:
        """Run hypervolume indicator selection."""
        population = []
        for _ in range(self.pop_size):
            indices = sorted(self.rng.choice(self.n_items, size=self.k, replace=False).tolist())
            obj = self._evaluate(indices)
            population.append(Solution(indices=indices, objectives=obj))

        ref_point = np.zeros(self.n_obj)

        for gen in range(n_generations):
            # Create offspring via mutation
            offspring = []
            for sol in population:
                new_indices = sol.indices.copy()
                pos = self.rng.randint(len(new_indices))
                available = [i for i in range(self.n_items) if i not in new_indices]
                if available:
                    new_indices[pos] = self.rng.choice(available)
                new_indices = sorted(new_indices)
                obj = self._evaluate(new_indices)
                offspring.append(Solution(indices=new_indices, objectives=obj))

            combined = population + offspring

            # Select based on hypervolume contribution
            selected: List[Solution] = []
            remaining = list(range(len(combined)))

            while len(selected) < self.pop_size and remaining:
                # Non-dominated sort first
                temp_sols = [combined[i] for i in remaining]
                fronts = non_dominated_sort(temp_sols)

                if len(selected) + len(fronts[0]) <= self.pop_size:
                    for idx in fronts[0]:
                        selected.append(combined[remaining[idx]])
                    remaining = [remaining[i] for i in range(len(remaining)) if i not in fronts[0]]
                else:
                    # Use hypervolume contribution
                    front_sols = [combined[remaining[i]] for i in fronts[0]]
                    contributions = []
                    for fi, sol in enumerate(front_sols):
                        others = [s for fj, s in enumerate(front_sols) if fj != fi]
                        hvc = self._hypervolume_contribution(sol, others, ref_point)
                        contributions.append((fi, hvc))
                    contributions.sort(key=lambda x: x[1], reverse=True)
                    needed = self.pop_size - len(selected)
                    for fi, _ in contributions[:needed]:
                        selected.append(front_sols[fi])
                    break

            population = selected

        fronts = non_dominated_sort(population)
        pareto = [population[i] for i in fronts[0]]

        return MOOResult(
            pareto_front=ParetoFront(solutions=pareto, n_objectives=self.n_obj),
            n_generations=n_generations,
            metadata={"algorithm": "hypervolume_selection"},
        )


# ---------------------------------------------------------------------------
# Scalarization Methods
# ---------------------------------------------------------------------------

class WeightedSumScalarization:
    """Weighted sum scalarization with learned/given weights."""

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        weights: Optional[np.ndarray] = None,
    ):
        self.objectives = objectives
        self.n_obj = len(objectives)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.n_obj) / self.n_obj

    def evaluate(self, indices: List[int]) -> float:
        obj_vals = np.array([obj.evaluate(indices) for obj in self.objectives])
        return float(self.weights @ obj_vals)

    def learn_weights(
        self,
        preference_pairs: List[Tuple[List[int], List[int]]],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> np.ndarray:
        """Learn weights from pairwise preferences.

        Given pairs (A, B) where A is preferred to B,
        adjust weights so that w^T f(A) > w^T f(B).
        """
        w = self.weights.copy()
        for _ in range(n_iterations):
            for A, B in preference_pairs:
                f_A = np.array([obj.evaluate(A) for obj in self.objectives])
                f_B = np.array([obj.evaluate(B) for obj in self.objectives])
                diff = f_A - f_B
                score = w @ diff
                if score < 0:
                    # Increase weights on objectives where A is better
                    w += learning_rate * diff
                    w = np.maximum(w, 0)
                    w_sum = np.sum(w)
                    if w_sum > 0:
                        w /= w_sum
        self.weights = w
        return w


class TchebycheffScalarization:
    """Tchebycheff scalarization for Pareto approximation.

    g(x|w, z*) = max_i w_i |f_i(x) - z*_i|
    """

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        weights: np.ndarray,
        ideal_point: Optional[np.ndarray] = None,
    ):
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.weights = weights
        self.ideal_point = ideal_point

    def set_ideal_point(self, solutions: List[List[int]]) -> None:
        """Estimate ideal point from a set of solutions."""
        obj_matrix = np.array([
            [obj.evaluate(s) for obj in self.objectives] for s in solutions
        ])
        self.ideal_point = np.max(obj_matrix, axis=0)

    def evaluate(self, indices: List[int]) -> float:
        """Evaluate (lower is better for Tchebycheff)."""
        obj_vals = np.array([obj.evaluate(indices) for obj in self.objectives])
        if self.ideal_point is None:
            self.ideal_point = obj_vals
        return float(np.max(self.weights * np.abs(obj_vals - self.ideal_point)))

    def optimize_greedy(
        self, n_items: int, k: int, seed: int = 42,
    ) -> Solution:
        """Greedy optimization of Tchebycheff scalarization."""
        rng = np.random.RandomState(seed)
        best_indices = sorted(rng.choice(n_items, size=k, replace=False).tolist())
        best_val = self.evaluate(best_indices)

        for _ in range(500):
            # Swap one item
            new_indices = best_indices.copy()
            pos = rng.randint(k)
            available = [i for i in range(n_items) if i not in new_indices]
            if not available:
                break
            new_indices[pos] = rng.choice(available)
            new_indices = sorted(new_indices)
            val = self.evaluate(new_indices)
            if val < best_val:
                best_val = val
                best_indices = new_indices

        obj_vals = np.array([obj.evaluate(best_indices) for obj in self.objectives])
        return Solution(indices=best_indices, objectives=obj_vals)


class AchievementScalarization:
    """Achievement scalarizing function (ASF).

    max_i (f_i(x) - z_i) / w_i + rho * sum_i f_i(x)
    """

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        weights: np.ndarray,
        aspiration: Optional[np.ndarray] = None,
        rho: float = 0.01,
    ):
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.weights = weights
        self.aspiration = aspiration if aspiration is not None else np.zeros(self.n_obj)
        self.rho = rho

    def evaluate(self, indices: List[int]) -> float:
        obj_vals = np.array([obj.evaluate(indices) for obj in self.objectives])
        weighted_gaps = (obj_vals - self.aspiration) / np.maximum(self.weights, 1e-12)
        return float(np.max(weighted_gaps) + self.rho * np.sum(obj_vals))


# ---------------------------------------------------------------------------
# Pareto Analysis
# ---------------------------------------------------------------------------

class ParetoAnalysis:
    """Analysis tools for Pareto fronts."""

    @staticmethod
    def marginal_rate_of_substitution(
        front: ParetoFront,
        obj_i: int = 0,
        obj_j: int = 1,
    ) -> List[float]:
        """Compute MRS along the Pareto front.

        MRS = -df_j / df_i at each point.
        """
        solutions = sorted(front.solutions, key=lambda s: s.objectives[obj_i])
        mrs_values = []
        for k in range(1, len(solutions)):
            d_fi = solutions[k].objectives[obj_i] - solutions[k - 1].objectives[obj_i]
            d_fj = solutions[k].objectives[obj_j] - solutions[k - 1].objectives[obj_j]
            if abs(d_fi) > 1e-12:
                mrs_values.append(-d_fj / d_fi)
            else:
                mrs_values.append(float("inf"))
        return mrs_values

    @staticmethod
    def knee_point_angle(front: ParetoFront) -> Optional[Solution]:
        """Find knee point using maximum angle method."""
        if len(front.solutions) < 3:
            return front.solutions[0] if front.solutions else None
        n_obj = front.solutions[0].objectives.shape[0]
        if n_obj != 2:
            return front.solutions[len(front.solutions) // 2]

        sorted_sols = sorted(front.solutions, key=lambda s: s.objectives[0])
        max_dist = 0.0
        knee = sorted_sols[0]

        # Line from first to last
        p1 = sorted_sols[0].objectives
        p2 = sorted_sols[-1].objectives
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-12:
            return sorted_sols[0]
        line_dir = line_vec / line_len

        for sol in sorted_sols[1:-1]:
            point_vec = sol.objectives - p1
            proj = np.dot(point_vec, line_dir)
            closest = p1 + proj * line_dir
            dist = np.linalg.norm(sol.objectives - closest)
            if dist > max_dist:
                max_dist = dist
                knee = sol

        return knee

    @staticmethod
    def spread_metric(front: ParetoFront) -> float:
        """Compute spread metric (uniformity of Pareto front)."""
        if len(front.solutions) < 2:
            return 0.0
        obj_matrix = np.array([s.objectives for s in front.solutions])
        sorted_by_first = obj_matrix[np.argsort(obj_matrix[:, 0])]

        # Consecutive distances
        distances = []
        for i in range(len(sorted_by_first) - 1):
            d = np.linalg.norm(sorted_by_first[i + 1] - sorted_by_first[i])
            distances.append(d)

        if len(distances) == 0:
            return 0.0
        mean_d = np.mean(distances)
        spread = sum(abs(d - mean_d) for d in distances) / (len(distances) * mean_d + 1e-12)
        return float(1.0 - spread)  # Higher = more uniform

    @staticmethod
    def generational_distance(
        front: ParetoFront,
        reference_front: List[np.ndarray],
    ) -> float:
        """Generational distance to a reference front."""
        if len(front.solutions) == 0 or len(reference_front) == 0:
            return float("inf")
        total = 0.0
        for sol in front.solutions:
            min_dist = min(
                np.linalg.norm(sol.objectives - ref) for ref in reference_front
            )
            total += min_dist ** 2
        return math.sqrt(total / len(front.solutions))


# ---------------------------------------------------------------------------
# MOO Comparison
# ---------------------------------------------------------------------------

class MOOComparison:
    """Compare multi-objective optimization algorithms."""

    def __init__(
        self,
        n_items: int,
        k: int,
        objectives: List[ObjectiveFunction],
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.objectives = objectives
        self.seed = seed

    def compare(
        self,
        n_generations: int = 50,
    ) -> Dict[str, MOOResult]:
        """Run and compare MOO algorithms."""
        results: Dict[str, MOOResult] = {}

        # NSGA-II
        nsga = NSGA2(
            self.n_items, self.k, self.objectives,
            population_size=30, seed=self.seed,
        )
        results["NSGA-II"] = nsga.run(n_generations)

        # MOEA/D
        moead = MOEAD(
            self.n_items, self.k, self.objectives,
            n_weight_vectors=30, seed=self.seed,
        )
        results["MOEA/D"] = moead.run(n_generations)

        # Hypervolume
        hvs = HypervolumeSelection(
            self.n_items, self.k, self.objectives,
            population_size=30, seed=self.seed,
        )
        results["HV-Selection"] = hvs.run(n_generations)

        return results


# ---------------------------------------------------------------------------
# Epsilon-Constraint Method
# ---------------------------------------------------------------------------

class EpsilonConstraint:
    """Epsilon-constraint method for Pareto frontier computation.

    Optimize one objective while constraining others to be above thresholds.
    By varying thresholds, trace out the Pareto front.
    """

    def __init__(
        self,
        n_items: int,
        k: int,
        objectives: List[ObjectiveFunction],
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.rng = np.random.RandomState(seed)

    def _evaluate(self, indices: List[int]) -> np.ndarray:
        return np.array([obj.evaluate(indices) for obj in self.objectives])

    def _optimize_single(
        self,
        primary_obj: int,
        constraints: Dict[int, float],
        n_trials: int = 200,
    ) -> Optional[Solution]:
        """Optimize primary objective subject to constraints on others."""
        best_val = -float("inf")
        best_indices = None

        for _ in range(n_trials):
            indices = sorted(self.rng.choice(self.n_items, self.k, replace=False).tolist())
            obj_vals = self._evaluate(indices)

            # Check constraints
            feasible = True
            for obj_idx, threshold in constraints.items():
                if obj_vals[obj_idx] < threshold:
                    feasible = False
                    break

            if feasible and obj_vals[primary_obj] > best_val:
                best_val = obj_vals[primary_obj]
                best_indices = indices

        if best_indices is None:
            return None
        return Solution(indices=best_indices, objectives=self._evaluate(best_indices))

    def compute_frontier(
        self,
        n_points: int = 20,
        n_trials: int = 200,
    ) -> ParetoFront:
        """Compute Pareto frontier via epsilon-constraint method."""
        solutions: List[Solution] = []

        # Find ranges for each objective
        obj_ranges: List[Tuple[float, float]] = []
        sample_vals = []
        for _ in range(100):
            indices = sorted(self.rng.choice(self.n_items, self.k, replace=False).tolist())
            vals = self._evaluate(indices)
            sample_vals.append(vals)
        sample_arr = np.array(sample_vals)

        for m in range(self.n_obj):
            obj_ranges.append((float(np.min(sample_arr[:, m])), float(np.max(sample_arr[:, m]))))

        # Vary constraint on second objective
        if self.n_obj >= 2:
            lo, hi = obj_ranges[1]
            thresholds = np.linspace(lo, hi, n_points)

            for threshold in thresholds:
                sol = self._optimize_single(0, {1: threshold}, n_trials)
                if sol is not None:
                    solutions.append(sol)

        # Filter dominated
        if solutions:
            fronts = non_dominated_sort(solutions)
            solutions = [solutions[i] for i in fronts[0]]

        return ParetoFront(solutions=solutions, n_objectives=self.n_obj)


# ---------------------------------------------------------------------------
# Weighted Tchebycheff Scan
# ---------------------------------------------------------------------------

class WeightedTchebycheffScan:
    """Scan the Pareto front by varying Tchebycheff weights."""

    def __init__(
        self,
        n_items: int,
        k: int,
        objectives: List[ObjectiveFunction],
        n_weight_points: int = 30,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n_points = n_weight_points
        self.rng = np.random.RandomState(seed)

    def scan(self, n_trials_per_weight: int = 100) -> ParetoFront:
        """Scan Pareto front via Tchebycheff scalarizations."""
        solutions: List[Solution] = []

        # Estimate ideal point
        ideal = np.zeros(self.n_obj)
        for _ in range(200):
            indices = sorted(self.rng.choice(self.n_items, self.k, replace=False).tolist())
            vals = np.array([obj.evaluate(indices) for obj in self.objectives])
            ideal = np.maximum(ideal, vals)

        # Generate weight vectors
        if self.n_obj == 2:
            weights_list = [
                np.array([i / (self.n_points - 1), 1 - i / (self.n_points - 1)])
                for i in range(self.n_points)
            ]
        else:
            weights_list = [
                self.rng.dirichlet(np.ones(self.n_obj))
                for _ in range(self.n_points)
            ]

        for weights in weights_list:
            te = TchebycheffScalarization(self.objectives, weights, ideal)
            sol = te.optimize_greedy(self.n_items, self.k)
            solutions.append(sol)

        # Filter dominated
        if solutions:
            fronts = non_dominated_sort(solutions)
            solutions = [solutions[i] for i in fronts[0]]

        return ParetoFront(solutions=solutions, n_objectives=self.n_obj)


# ---------------------------------------------------------------------------
# Quality-Diversity MAP-Elites
# ---------------------------------------------------------------------------

class MAPElites:
    """MAP-Elites for quality-diversity optimization.

    Maintains an archive of solutions binned by behavior characteristics.
    Each cell contains the highest-quality solution with that behavior.
    """

    def __init__(
        self,
        n_items: int,
        k: int,
        quality_fn: Callable[[List[int]], float],
        behavior_fn: Callable[[List[int]], np.ndarray],
        n_bins_per_dim: int = 10,
        behavior_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: int = 42,
    ):
        self.n_items = n_items
        self.k = k
        self.quality_fn = quality_fn
        self.behavior_fn = behavior_fn
        self.n_bins = n_bins_per_dim
        self.rng = np.random.RandomState(seed)

        # Behavior bounds
        if behavior_bounds is not None:
            self.b_lo, self.b_hi = behavior_bounds
        else:
            self.b_lo = np.zeros(2)
            self.b_hi = np.ones(2)

        self.archive: Dict[Tuple[int, ...], Tuple[List[int], float]] = {}

    def _to_bin(self, behavior: np.ndarray) -> Tuple[int, ...]:
        """Convert behavior vector to bin index."""
        normalized = (behavior - self.b_lo) / (self.b_hi - self.b_lo + 1e-12)
        normalized = np.clip(normalized, 0, 0.999)
        bins = (normalized * self.n_bins).astype(int)
        return tuple(bins.tolist())

    def _random_solution(self) -> List[int]:
        return sorted(self.rng.choice(self.n_items, self.k, replace=False).tolist())

    def _mutate(self, solution: List[int]) -> List[int]:
        new = solution.copy()
        pos = self.rng.randint(len(new))
        available = [i for i in range(self.n_items) if i not in new]
        if available:
            new[pos] = self.rng.choice(available)
        return sorted(new)

    def run(self, n_iterations: int = 1000) -> Dict[Tuple[int, ...], Tuple[List[int], float]]:
        """Run MAP-Elites."""
        # Initialize with random solutions
        for _ in range(min(100, n_iterations)):
            sol = self._random_solution()
            quality = self.quality_fn(sol)
            behavior = self.behavior_fn(sol)
            bin_idx = self._to_bin(behavior)

            if bin_idx not in self.archive or quality > self.archive[bin_idx][1]:
                self.archive[bin_idx] = (sol, quality)

        # Mutation phase
        for _ in range(max(0, n_iterations - 100)):
            if not self.archive:
                continue
            # Select random solution from archive
            key = self.rng.choice(list(self.archive.keys()))
            parent = self.archive[key][0]
            child = self._mutate(parent)

            quality = self.quality_fn(child)
            behavior = self.behavior_fn(child)
            bin_idx = self._to_bin(behavior)

            if bin_idx not in self.archive or quality > self.archive[bin_idx][1]:
                self.archive[bin_idx] = (child, quality)

        return self.archive

    def coverage(self) -> float:
        """Fraction of bins filled."""
        total_bins = self.n_bins ** len(self.b_lo)
        return len(self.archive) / total_bins

    def qd_score(self) -> float:
        """Sum of qualities across all bins (QD-Score)."""
        return sum(q for _, q in self.archive.values())
