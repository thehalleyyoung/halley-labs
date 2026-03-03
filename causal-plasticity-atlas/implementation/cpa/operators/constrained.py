"""Constraint-respecting genetic operators.

Ensures that mutations and crossovers honour user-specified edge
constraints (required, forbidden), degree bounds, and optional
tier orderings.

Classes
-------
EdgeConstraints
    Declarative constraint specification.
ConstrainedOperator
    Wraps any operator to enforce EdgeConstraints.
ConstrainedMutation
    Mutation wrapper that projects results to feasible space.
ConstrainedCrossover
    Crossover wrapper that repairs constraint violations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_acyclic(adj: NDArray) -> bool:
    """Return True if *adj* is a DAG (Kahn's algorithm)."""
    n = adj.shape[0]
    in_deg = (adj != 0).sum(axis=0).copy()
    queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in range(n):
            if adj[u, v] != 0:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    return visited == n


def _has_path(adj: NDArray, source: int, target: int) -> bool:
    """Return True if there is a directed path from *source* to *target*."""
    n = adj.shape[0]
    visited: Set[int] = set()
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == target:
            return True
        for v in range(n):
            if adj[u, v] != 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    return False


def _existing_edges(adj: NDArray) -> List[Tuple[int, int]]:
    """Return list of (i, j) with adj[i, j] != 0."""
    rows, cols = np.nonzero(adj)
    return list(zip(rows.tolist(), cols.tolist()))


def _repair_dag(adj: NDArray) -> NDArray:
    """Remove back-edges to ensure acyclicity."""
    result = adj.copy()
    if _is_acyclic(result):
        return result
    n = result.shape[0]
    in_deg = (result != 0).sum(axis=0).copy()
    order: List[int] = []
    remaining = set(range(n))
    while remaining:
        best = min(remaining, key=lambda x: in_deg[x])
        order.append(best)
        remaining.remove(best)
        for v in range(n):
            if result[best, v] != 0 and v in remaining:
                in_deg[v] -= 1
    pos = {v: i for i, v in enumerate(order)}
    for i in range(n):
        for j in range(n):
            if result[i, j] != 0 and pos[i] >= pos[j]:
                result[i, j] = 0.0
    return result


# ------------------------------------------------------------------
# EdgeConstraints
# ------------------------------------------------------------------


@dataclass
class EdgeConstraints:
    """Known structural constraints on the DAG.

    Attributes
    ----------
    required_edges : Set[Tuple[int, int]]
        Edges that must appear in every valid DAG.
    forbidden_edges : Set[Tuple[int, int]]
        Edges that must never appear.
    tier_ordering : Optional[List[List[int]]]
        Tier ordering: edges may only go from earlier to later tiers.
    max_parents : Optional[int]
        Maximum number of parents per node.
    max_degree : Optional[int]
        Maximum total degree (in + out) per node.
    """

    required_edges: Set[Tuple[int, int]] = field(default_factory=set)
    forbidden_edges: Set[Tuple[int, int]] = field(default_factory=set)
    tier_ordering: Optional[List[List[int]]] = None
    max_parents: Optional[int] = None
    max_degree: Optional[int] = None

    def is_valid(self, adj: NDArray) -> bool:
        """Check if DAG satisfies all constraints."""
        return (
            self._check_required(adj)
            and self._check_forbidden(adj)
            and self._check_max_parents(adj)
            and self._check_max_degree(adj)
            and self._check_tier_ordering(adj)
            and _is_acyclic(adj)
        )

    def _check_required(self, adj: NDArray) -> bool:
        """All required edges present."""
        for i, j in self.required_edges:
            if adj[i, j] == 0:
                return False
        return True

    def _check_forbidden(self, adj: NDArray) -> bool:
        """No forbidden edges present."""
        for i, j in self.forbidden_edges:
            if adj[i, j] != 0:
                return False
        return True

    def _check_max_parents(self, adj: NDArray) -> bool:
        """No node exceeds max parents."""
        if self.max_parents is None:
            return True
        in_deg = (adj != 0).sum(axis=0)
        return bool(np.all(in_deg <= self.max_parents))

    def _check_max_degree(self, adj: NDArray) -> bool:
        """No node exceeds max degree."""
        if self.max_degree is None:
            return True
        n = adj.shape[0]
        binary = (adj != 0).astype(int)
        degree = binary.sum(axis=0) + binary.sum(axis=1)
        return bool(np.all(degree <= self.max_degree))

    def _check_tier_ordering(self, adj: NDArray) -> bool:
        """Edges only go from earlier tiers to later tiers."""
        if self.tier_ordering is None:
            return True
        node_tier = {}
        for tier_idx, tier_nodes in enumerate(self.tier_ordering):
            for node in tier_nodes:
                node_tier[node] = tier_idx
        n = adj.shape[0]
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0:
                    ti = node_tier.get(i, -1)
                    tj = node_tier.get(j, -1)
                    if ti >= 0 and tj >= 0 and ti >= tj:
                        return False
        return True

    def valid_additions(self, adj: NDArray) -> List[Tuple[int, int]]:
        """Edges that can be added without violating constraints."""
        n = adj.shape[0]
        binary = (adj != 0).astype(int)
        in_deg = binary.sum(axis=0)
        degree = binary.sum(axis=0) + binary.sum(axis=1)
        candidates: List[Tuple[int, int]] = []

        node_tier = {}
        if self.tier_ordering is not None:
            for tier_idx, tier_nodes in enumerate(self.tier_ordering):
                for node in tier_nodes:
                    node_tier[node] = tier_idx

        for i in range(n):
            for j in range(n):
                if i == j or adj[i, j] != 0:
                    continue
                if (i, j) in self.forbidden_edges:
                    continue
                if self.max_parents is not None and in_deg[j] >= self.max_parents:
                    continue
                if self.max_degree is not None:
                    if degree[i] >= self.max_degree or degree[j] >= self.max_degree:
                        continue
                if node_tier:
                    ti = node_tier.get(i, -1)
                    tj = node_tier.get(j, -1)
                    if ti >= 0 and tj >= 0 and ti >= tj:
                        continue
                if not _has_path(adj, j, i):
                    candidates.append((i, j))
        return candidates

    def valid_removals(self, adj: NDArray) -> List[Tuple[int, int]]:
        """Edges that can be removed without violating constraints."""
        edges = _existing_edges(adj)
        return [(i, j) for i, j in edges if (i, j) not in self.required_edges]


# ------------------------------------------------------------------
# ConstrainedOperator
# ------------------------------------------------------------------


class ConstrainedOperator:
    """Genetic operator wrapper that enforces :class:`EdgeConstraints`.

    Wraps any base operator (mutation or crossover) and projects the
    result onto the feasible set defined by the constraints.

    Parameters
    ----------
    constraints : EdgeConstraints
        The structural constraints to enforce.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        constraints: EdgeConstraints,
        seed: Optional[int] = None,
    ) -> None:
        self.constraints = constraints
        self._rng = np.random.default_rng(seed)

    def is_valid(self, dag: NDArray) -> bool:
        """Return ``True`` if *dag* satisfies all constraints."""
        return self.constraints.is_valid(dag)

    def project(self, dag: NDArray) -> NDArray:
        """Project *dag* onto the nearest constraint-satisfying DAG."""
        result = dag.copy()

        # 1. Remove forbidden edges
        for i, j in self.constraints.forbidden_edges:
            if i < result.shape[0] and j < result.shape[1]:
                result[i, j] = 0.0

        # 2. Add required edges
        for i, j in self.constraints.required_edges:
            if i < result.shape[0] and j < result.shape[1]:
                if result[i, j] == 0:
                    result[i, j] = 1.0

        # 3. Enforce tier ordering
        if self.constraints.tier_ordering is not None:
            node_tier = {}
            for tier_idx, tier_nodes in enumerate(self.constraints.tier_ordering):
                for node in tier_nodes:
                    node_tier[node] = tier_idx
            n = result.shape[0]
            for i in range(n):
                for j in range(n):
                    if result[i, j] != 0:
                        ti = node_tier.get(i, -1)
                        tj = node_tier.get(j, -1)
                        if ti >= 0 and tj >= 0 and ti >= tj:
                            result[i, j] = 0.0

        # 4. Enforce max parents by removing weakest excess edges
        if self.constraints.max_parents is not None:
            n = result.shape[0]
            for j in range(n):
                parents = [(i, abs(result[i, j])) for i in range(n) if result[i, j] != 0]
                if len(parents) > self.constraints.max_parents:
                    # Keep strongest, remove weakest (but keep required)
                    parents.sort(key=lambda x: x[1])
                    n_remove = len(parents) - self.constraints.max_parents
                    removed = 0
                    for i, _ in parents:
                        if removed >= n_remove:
                            break
                        if (i, j) not in self.constraints.required_edges:
                            result[i, j] = 0.0
                            removed += 1

        # 5. Enforce max degree similarly
        if self.constraints.max_degree is not None:
            n = result.shape[0]
            binary = (result != 0).astype(int)
            degree = binary.sum(axis=0) + binary.sum(axis=1)
            for node in range(n):
                while degree[node] > self.constraints.max_degree:
                    # Remove weakest non-required edge
                    edges = []
                    for j in range(n):
                        if result[node, j] != 0 and (node, j) not in self.constraints.required_edges:
                            edges.append((node, j, abs(result[node, j])))
                    for i in range(n):
                        if result[i, node] != 0 and (i, node) not in self.constraints.required_edges:
                            edges.append((i, node, abs(result[i, node])))
                    if not edges:
                        break
                    edges.sort(key=lambda x: x[2])
                    ei, ej, _ = edges[0]
                    result[ei, ej] = 0.0
                    binary = (result != 0).astype(int)
                    degree = binary.sum(axis=0) + binary.sum(axis=1)

        # 6. Ensure acyclicity
        result = _repair_dag(result)

        # 7. Re-add required edges that may have been removed by DAG repair
        for i, j in self.constraints.required_edges:
            if i < result.shape[0] and j < result.shape[1]:
                result[i, j] = dag[i, j] if dag[i, j] != 0 else 1.0

        return result

    def constrained_mutation(
        self,
        dag: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Mutate *dag* while respecting constraints."""
        rng = rng or self._rng
        result = dag.copy()
        n = result.shape[0]

        op = int(rng.integers(0, 3))
        if op == 0:
            candidates = self.constraints.valid_additions(result)
            if candidates:
                idx = int(rng.integers(0, len(candidates)))
                i, j = candidates[idx]
                result[i, j] = 1.0
        elif op == 1:
            candidates = self.constraints.valid_removals(result)
            if candidates:
                idx = int(rng.integers(0, len(candidates)))
                i, j = candidates[idx]
                result[i, j] = 0.0
        else:
            removable = self.constraints.valid_removals(result)
            if removable:
                rng.shuffle(removable)  # type: ignore[arg-type]
                for i, j in removable:
                    w = result[i, j]
                    result[i, j] = 0.0
                    if (j, i) not in self.constraints.forbidden_edges:
                        if not _has_path(result, j, i):
                            result[j, i] = w
                            if self.constraints.is_valid(result):
                                return result
                            result[j, i] = 0.0
                    result[i, j] = w

        if not self.constraints.is_valid(result):
            result = self.project(result)
        return result

    def constrained_crossover(
        self,
        dag1: NDArray,
        dag2: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Crossover two DAGs while respecting constraints."""
        rng = rng or self._rng
        n = dag1.shape[0]
        mask = rng.random((n, n)) < 0.5
        offspring = np.where(mask, dag1, dag2)
        np.fill_diagonal(offspring, 0.0)
        offspring = self.project(offspring)
        return offspring


# ------------------------------------------------------------------
# ConstrainedMutation
# ------------------------------------------------------------------


class ConstrainedMutation:
    """Mutation wrapper that respects constraints.

    Applies a base mutation operator then projects the result
    to the feasible set.

    Parameters
    ----------
    constraints : EdgeConstraints
        Constraints to enforce.
    base_mutation : object
        Any object with a ``mutate(adj, rng)`` method.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        constraints: EdgeConstraints,
        base_mutation: object,
        seed: Optional[int] = None,
    ) -> None:
        self.constraints = constraints
        self.base_mutation = base_mutation
        self._projector = ConstrainedOperator(constraints, seed=seed)
        self._rng = np.random.default_rng(seed)

    def mutate(
        self,
        adj: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Mutate respecting constraints."""
        rng = rng or self._rng
        mutated = self.base_mutation.mutate(adj, rng)  # type: ignore[union-attr]
        return self._project_to_feasible(mutated)

    def _project_to_feasible(self, adj: NDArray) -> NDArray:
        """Project DAG to nearest feasible DAG."""
        return self._projector.project(adj)


# ------------------------------------------------------------------
# ConstrainedCrossover
# ------------------------------------------------------------------


class ConstrainedCrossover:
    """Crossover wrapper that repairs constraint violations.

    Parameters
    ----------
    constraints : EdgeConstraints
        Constraints to enforce.
    base_crossover : object
        Any object with a ``crossover(p1, p2, rng)`` method.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        constraints: EdgeConstraints,
        base_crossover: object,
        seed: Optional[int] = None,
    ) -> None:
        self.constraints = constraints
        self.base_crossover = base_crossover
        self._projector = ConstrainedOperator(constraints, seed=seed)
        self._rng = np.random.default_rng(seed)

    def crossover(
        self,
        parent1: NDArray,
        parent2: NDArray,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Crossover respecting constraints."""
        rng = rng or self._rng
        offspring = self.base_crossover.crossover(parent1, parent2, rng)  # type: ignore[union-attr]
        return self._repair_constraints(offspring)

    def _repair_constraints(self, adj: NDArray) -> NDArray:
        """Repair constraint violations."""
        return self._projector.project(adj)
