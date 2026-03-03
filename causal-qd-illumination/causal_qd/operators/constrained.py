"""Constraint-aware mutation and crossover operators.

Wraps existing operators to enforce structural constraints such as
forbidden/required edges, max-parent limits, and temporal tier orderings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from causal_qd.types import AdjacencyMatrix
from causal_qd.operators.mutation import MutationOperator, _has_cycle, _topological_sort
from causal_qd.operators.crossover import CrossoverOperator, _break_cycles_with_ordering


# ---------------------------------------------------------------------------
# TierConstraints
# ---------------------------------------------------------------------------

class TierConstraints:
    """Temporal tier ordering for nodes.

    Nodes are grouped into ordered tiers.  Edges may only go from
    earlier tiers to later tiers (never within or backward).

    Parameters
    ----------
    tiers : list[set[int]]
        Ordered groups of node indices.
    """

    def __init__(self, tiers: list[set[int]]) -> None:
        self.tiers = tiers
        self._node_tier: dict[int, int] = {}
        for tier_idx, tier in enumerate(tiers):
            for node in tier:
                self._node_tier[node] = tier_idx

    def is_valid_edge(self, i: int, j: int) -> bool:
        """Return True if edge i→j respects tier ordering."""
        ti = self._node_tier.get(i)
        tj = self._node_tier.get(j)
        if ti is None or tj is None:
            return True  # nodes not in any tier are unconstrained
        return ti < tj

    def to_edge_constraints(self) -> EdgeConstraints:
        """Convert tier ordering to an EdgeConstraints with forbidden edges."""
        forbidden: set[tuple[int, int]] = set()
        all_nodes = [n for tier in self.tiers for n in tier]
        for i in all_nodes:
            for j in all_nodes:
                if i != j and not self.is_valid_edge(i, j):
                    forbidden.add((i, j))
        return EdgeConstraints(forbidden_edges=frozenset(forbidden))


# ---------------------------------------------------------------------------
# EdgeConstraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeConstraints:
    """Structural constraints on DAG edges.

    Parameters
    ----------
    forbidden_edges : frozenset[tuple[int, int]]
        Edges that must NOT be present.
    required_edges : frozenset[tuple[int, int]]
        Edges that MUST be present.
    max_parents : int | None
        Maximum number of parents per node (None = no limit).
    tier_ordering : list[set[int]] | None
        Temporal tier ordering (edges only go forward between tiers).
    """

    forbidden_edges: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    required_edges: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    max_parents: Optional[int] = None
    tier_ordering: Optional[list[set[int]]] = None

    def validate(self) -> None:
        """Check constraint consistency.

        Raises
        ------
        ValueError
            If required and forbidden edges conflict, or tier ordering
            conflicts with required edges.
        """
        conflicts = self.required_edges & self.forbidden_edges
        if conflicts:
            raise ValueError(
                f"Edges cannot be both required and forbidden: {conflicts}"
            )

        if self.tier_ordering is not None:
            tc = TierConstraints(self.tier_ordering)
            for i, j in self.required_edges:
                if not tc.is_valid_edge(i, j):
                    raise ValueError(
                        f"Required edge ({i}, {j}) violates tier ordering."
                    )

    def is_valid_edge(self, i: int, j: int, dag: AdjacencyMatrix) -> bool:
        """Check if adding edge i→j is valid under all constraints."""
        if (i, j) in self.forbidden_edges:
            return False

        if self.tier_ordering is not None:
            tc = TierConstraints(self.tier_ordering)
            if not tc.is_valid_edge(i, j):
                return False

        if self.max_parents is not None:
            current_parents = int(dag[:, j].sum())
            if not dag[i, j] and current_parents >= self.max_parents:
                return False

        return True

    def is_valid_dag(self, dag: AdjacencyMatrix) -> bool:
        """Check if DAG satisfies all constraints."""
        n = dag.shape[0]

        # Check forbidden edges
        for i, j in self.forbidden_edges:
            if i < n and j < n and dag[i, j]:
                return False

        # Check required edges
        for i, j in self.required_edges:
            if i < n and j < n and not dag[i, j]:
                return False

        # Check max parents
        if self.max_parents is not None:
            for j in range(n):
                if int(dag[:, j].sum()) > self.max_parents:
                    return False

        # Check tier ordering
        if self.tier_ordering is not None:
            tc = TierConstraints(self.tier_ordering)
            for i in range(n):
                for j in range(n):
                    if dag[i, j] and not tc.is_valid_edge(i, j):
                        return False

        return True


# ---------------------------------------------------------------------------
# Repair helper
# ---------------------------------------------------------------------------

def _repair(dag: AdjacencyMatrix, constraints: EdgeConstraints) -> AdjacencyMatrix:
    """Repair constraint violations in *dag*.

    1. Remove forbidden edges.
    2. Remove tier-violating edges.
    3. Add required edges (skip if would create cycle).
    4. Remove excess parents (drop lowest-weight / last-added edges first).
    """
    result = dag.copy()
    n = result.shape[0]

    # 1. Remove forbidden edges
    for i, j in constraints.forbidden_edges:
        if i < n and j < n:
            result[i, j] = 0

    # 2. Remove tier-violating edges
    if constraints.tier_ordering is not None:
        tc = TierConstraints(constraints.tier_ordering)
        for i in range(n):
            for j in range(n):
                if result[i, j] and not tc.is_valid_edge(i, j):
                    result[i, j] = 0

    # 3. Add required edges (checking acyclicity)
    for i, j in constraints.required_edges:
        if i < n and j < n and not result[i, j]:
            result[i, j] = 1
            if _has_cycle(result):
                result[i, j] = 0

    # 4. Enforce max parents
    if constraints.max_parents is not None:
        for j in range(n):
            parents = list(np.nonzero(result[:, j])[0])
            if len(parents) > constraints.max_parents:
                # Keep required parents, remove others
                required_parents = [
                    p for p in parents if (int(p), j) in constraints.required_edges
                ]
                removable = [p for p in parents if p not in required_parents]
                # Remove from last to first (arbitrary but deterministic)
                while len(required_parents) + len(removable) > constraints.max_parents and removable:
                    p = removable.pop()
                    result[p, j] = 0

    return result


# ---------------------------------------------------------------------------
# ConstrainedMutation
# ---------------------------------------------------------------------------

class ConstrainedMutation(MutationOperator):
    """Wraps an existing mutation operator to enforce edge constraints.

    Applies the inner mutation, then repairs any constraint violations.
    If repair fails to produce a valid DAG, retries with fresh mutations.

    Parameters
    ----------
    inner : MutationOperator
        The underlying mutation operator.
    constraints : EdgeConstraints
        Structural constraints to enforce.
    max_retries : int
        Maximum number of mutation attempts before returning the
        repaired input (default 10).
    """

    def __init__(
        self,
        inner: MutationOperator,
        constraints: EdgeConstraints,
        max_retries: int = 10,
    ) -> None:
        self.inner = inner
        self.constraints = constraints
        self.max_retries = max_retries

    def mutate(self, dag: AdjacencyMatrix, rng: np.random.Generator) -> AdjacencyMatrix:
        """Apply inner mutation then repair constraint violations."""
        for _ in range(self.max_retries):
            candidate = self.inner.mutate(dag, rng)
            repaired = _repair(candidate, self.constraints)
            if not _has_cycle(repaired) and self.constraints.is_valid_dag(repaired):
                return repaired

        # Fallback: repair the original input
        return _repair(dag.copy(), self.constraints)


# ---------------------------------------------------------------------------
# ConstrainedCrossover
# ---------------------------------------------------------------------------

class ConstrainedCrossover(CrossoverOperator):
    """Wraps an existing crossover operator to enforce edge constraints.

    Applies the inner crossover, then repairs any constraint violations
    on both offspring.

    Parameters
    ----------
    inner : CrossoverOperator
        The underlying crossover operator.
    constraints : EdgeConstraints
        Structural constraints to enforce.
    max_retries : int
        Maximum number of crossover attempts before returning
        repaired inputs (default 10).
    """

    def __init__(
        self,
        inner: CrossoverOperator,
        constraints: EdgeConstraints,
        max_retries: int = 10,
    ) -> None:
        self.inner = inner
        self.constraints = constraints
        self.max_retries = max_retries

    def crossover(
        self,
        parent1: AdjacencyMatrix,
        parent2: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> Tuple[AdjacencyMatrix, AdjacencyMatrix]:
        """Apply inner crossover then repair constraint violations."""
        for _ in range(self.max_retries):
            child1, child2 = self.inner.crossover(parent1, parent2, rng)
            r1 = _repair(child1, self.constraints)
            r2 = _repair(child2, self.constraints)
            if (
                not _has_cycle(r1)
                and not _has_cycle(r2)
                and self.constraints.is_valid_dag(r1)
                and self.constraints.is_valid_dag(r2)
            ):
                return r1, r2

        # Fallback: repair the parents
        return (
            _repair(parent1.copy(), self.constraints),
            _repair(parent2.copy(), self.constraints),
        )
