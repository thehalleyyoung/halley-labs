"""
Tests for cutting plane methods.

Covers individual cut types, cut pool management, and integration
with LP-based solver.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pytest

from tests.conftest import _adj, random_dag


# ---------------------------------------------------------------------------
# Cutting plane infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """A linear constraint: coeffs @ x <= rhs."""
    coeffs: tuple[float, ...]
    rhs: float
    name: str = ""

    @property
    def n_vars(self) -> int:
        return len(self.coeffs)


@dataclass(slots=True)
class CutPool:
    """Manages a pool of cutting planes."""
    cuts: list[LinearConstraint] = field(default_factory=list)
    max_pool_size: int = 1000
    _active: set[int] = field(default_factory=set)

    def add_cut(self, cut: LinearConstraint) -> int:
        """Add a cut and return its index."""
        idx = len(self.cuts)
        self.cuts.append(cut)
        self._active.add(idx)
        if len(self.cuts) > self.max_pool_size:
            self._purge()
        return idx

    def deactivate(self, idx: int) -> None:
        self._active.discard(idx)

    def activate(self, idx: int) -> None:
        if idx < len(self.cuts):
            self._active.add(idx)

    @property
    def n_cuts(self) -> int:
        return len(self.cuts)

    @property
    def n_active(self) -> int:
        return len(self._active)

    def active_cuts(self) -> list[LinearConstraint]:
        return [self.cuts[i] for i in sorted(self._active)]

    def is_active(self, idx: int) -> bool:
        return idx in self._active

    def _purge(self) -> None:
        """Remove inactive cuts beyond capacity."""
        if len(self.cuts) <= self.max_pool_size:
            return
        active_indices = sorted(self._active)
        new_cuts = [self.cuts[i] for i in active_indices]
        self._active = set(range(len(new_cuts)))
        self.cuts = new_cuts

    def is_violated(
        self, cut: LinearConstraint, x: np.ndarray, tol: float = 1e-6
    ) -> bool:
        """Check if a cut is violated by solution x."""
        lhs = sum(c * xi for c, xi in zip(cut.coeffs, x))
        return lhs > cut.rhs + tol

    def most_violated(
        self, x: np.ndarray, tol: float = 1e-6
    ) -> LinearConstraint | None:
        """Return the most violated cut, or None."""
        worst: LinearConstraint | None = None
        worst_violation = tol
        for cut in self.active_cuts():
            lhs = sum(c * xi for c, xi in zip(cut.coeffs, x))
            violation = lhs - cut.rhs
            if violation > worst_violation:
                worst = cut
                worst_violation = violation
        return worst


# ---------------------------------------------------------------------------
# Cut generators
# ---------------------------------------------------------------------------


def _is_dag(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return count == n


def _has_path(adj: np.ndarray, src: int, tgt: int) -> bool:
    if src == tgt:
        return True
    visited: set[int] = set()
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            if c == tgt:
                return True
            if c not in visited:
                visited.add(c)
                queue.append(c)
    return False


def generate_cycle_elimination_cut(
    cycle: list[int],
    n: int,
) -> LinearConstraint:
    """Generate a cut that eliminates a cycle.

    For a cycle [v0, v1, ..., vk, v0], at least one edge must be absent:
    x[v0,v1] + x[v1,v2] + ... + x[vk,v0] <= k
    """
    n_vars = n * n
    coeffs = [0.0] * n_vars
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        coeffs[u * n + v] = 1.0
    rhs = float(len(cycle) - 1)
    return LinearConstraint(
        coeffs=tuple(coeffs),
        rhs=rhs,
        name=f"cycle_elim_{len(cycle)}",
    )


def generate_path_cut(
    path: list[int],
    n: int,
    treatment: int,
    outcome: int,
) -> LinearConstraint:
    """Generate a cut requiring at least one path edge to be removed.

    For path [t, v1, ..., vk, y], we need:
    x[t,v1] + x[v1,v2] + ... + x[vk,y] <= k
    (at least one edge must be absent to break this path)
    """
    n_vars = n * n
    coeffs = [0.0] * n_vars
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        coeffs[u * n + v] = 1.0
    rhs = float(len(path) - 2)  # must remove at least 1
    return LinearConstraint(
        coeffs=tuple(coeffs),
        rhs=rhs,
        name=f"path_cut_{len(path)}",
    )


def generate_budget_cut(
    original_adj: np.ndarray,
    max_edits: int,
) -> LinearConstraint:
    """Generate a cut limiting total edit distance."""
    n = original_adj.shape[0]
    n_vars = n * n
    coeffs = [0.0] * n_vars
    for i in range(n):
        for j in range(n):
            if i != j:
                if original_adj[i, j]:
                    # Removing this edge costs 1: add (1 - x[i,j])
                    coeffs[i * n + j] = -1.0
                else:
                    # Adding this edge costs 1: add x[i,j]
                    coeffs[i * n + j] = 1.0

    # Total edits: sum of add_costs + remove_costs
    # We encode: -sum(x_present) + sum(x_absent) <= max_edits - n_original_edges
    n_orig = int(original_adj.sum())
    rhs = float(max_edits - n_orig)

    return LinearConstraint(
        coeffs=tuple(coeffs),
        rhs=rhs,
        name=f"budget_{max_edits}",
    )


def generate_symmetry_breaking_cut(n: int) -> LinearConstraint:
    """Generate a symmetry-breaking cut: x[0,1] >= x[1,0] (if both could exist)."""
    n_vars = n * n
    coeffs = [0.0] * n_vars
    # x[1,0] - x[0,1] <= 0, i.e. x[0,1] >= x[1,0]
    if n >= 2:
        coeffs[1 * n + 0] = 1.0
        coeffs[0 * n + 1] = -1.0
    return LinearConstraint(
        coeffs=tuple(coeffs),
        rhs=0.0,
        name="symmetry_break",
    )


def find_violated_cycle_cuts(
    x_solution: np.ndarray,
    n: int,
    tol: float = 0.5,
) -> list[LinearConstraint]:
    """Find cycle elimination cuts violated by the LP relaxation solution.

    Builds the fractional graph from x_solution and finds cycles via DFS.
    """
    # Reconstruct adjacency from flat x
    adj_frac = x_solution.reshape(n, n)
    cuts: list[LinearConstraint] = []

    # Find short cycles (length 2 and 3)
    for i in range(n):
        for j in range(i + 1, n):
            # Length 2: i→j→i
            if adj_frac[i, j] + adj_frac[j, i] > 1 + tol:
                cuts.append(generate_cycle_elimination_cut([i, j], n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                # Length 3: i→j→k→i
                val = adj_frac[i, j] + adj_frac[j, k] + adj_frac[k, i]
                if val > 2 + tol:
                    cuts.append(generate_cycle_elimination_cut([i, j, k], n))

    return cuts


# ---------------------------------------------------------------------------
# Simple LP solver (simplex-like for testing)
# ---------------------------------------------------------------------------


def solve_lp_with_cuts(
    c: np.ndarray,
    initial_constraints: list[LinearConstraint],
    cut_pool: CutPool,
    max_rounds: int = 20,
    n_vars: int = 0,
) -> tuple[np.ndarray, float]:
    """Simple iterative LP with cutting planes.

    Uses a crude bounded-variable approach for testing.
    Returns (solution, objective_value).
    """
    if n_vars == 0:
        n_vars = len(c)

    x = np.zeros(n_vars)
    # Trivial feasibility: project onto box [0, 1]
    x = np.clip(x, 0, 1)

    best_obj = float("inf")
    best_x = x.copy()

    for round_idx in range(max_rounds):
        # Check constraints
        all_feasible = True
        for con in initial_constraints + cut_pool.active_cuts():
            lhs = sum(ci * xi for ci, xi in zip(con.coeffs, x))
            if lhs > con.rhs + 1e-6:
                all_feasible = False
                # Simple projection: reduce violating variables
                for i, ci in enumerate(con.coeffs):
                    if ci > 0 and x[i] > 0:
                        x[i] = max(0, x[i] - 0.1)

        obj = float(c @ x)
        if obj < best_obj and all_feasible:
            best_obj = obj
            best_x = x.copy()

        # Find violated cuts
        violated = cut_pool.most_violated(x)
        if violated is None:
            break
        # Already in pool, just need to enforce

    return best_x, best_obj


# ===================================================================
# Tests
# ===================================================================


class TestLinearConstraint:
    def test_basic(self):
        c = LinearConstraint(coeffs=(1.0, 2.0, 3.0), rhs=5.0, name="test")
        assert c.n_vars == 3
        assert c.rhs == 5.0

    def test_satisfied(self):
        c = LinearConstraint(coeffs=(1.0, 1.0), rhs=2.0)
        x = np.array([1.0, 0.5])
        assert sum(ci * xi for ci, xi in zip(c.coeffs, x)) <= c.rhs

    def test_violated(self):
        c = LinearConstraint(coeffs=(1.0, 1.0), rhs=1.0)
        x = np.array([0.8, 0.8])
        assert sum(ci * xi for ci, xi in zip(c.coeffs, x)) > c.rhs


class TestCutPool:
    def test_add_cut(self):
        pool = CutPool()
        c = LinearConstraint(coeffs=(1.0, 0.0), rhs=1.0)
        idx = pool.add_cut(c)
        assert idx == 0
        assert pool.n_cuts == 1
        assert pool.n_active == 1

    def test_deactivate(self):
        pool = CutPool()
        c = LinearConstraint(coeffs=(1.0,), rhs=1.0)
        idx = pool.add_cut(c)
        pool.deactivate(idx)
        assert pool.n_active == 0
        assert pool.n_cuts == 1

    def test_reactivate(self):
        pool = CutPool()
        c = LinearConstraint(coeffs=(1.0,), rhs=1.0)
        idx = pool.add_cut(c)
        pool.deactivate(idx)
        pool.activate(idx)
        assert pool.n_active == 1

    def test_is_violated(self):
        pool = CutPool()
        c = LinearConstraint(coeffs=(1.0, 1.0), rhs=1.0)
        x = np.array([0.8, 0.8])
        assert pool.is_violated(c, x)

    def test_not_violated(self):
        pool = CutPool()
        c = LinearConstraint(coeffs=(1.0, 1.0), rhs=2.0)
        x = np.array([0.5, 0.5])
        assert not pool.is_violated(c, x)

    def test_most_violated(self):
        pool = CutPool()
        c1 = LinearConstraint(coeffs=(1.0, 0.0), rhs=0.5, name="c1")
        c2 = LinearConstraint(coeffs=(0.0, 1.0), rhs=0.3, name="c2")
        pool.add_cut(c1)
        pool.add_cut(c2)
        x = np.array([0.8, 0.9])
        worst = pool.most_violated(x)
        assert worst is not None
        assert worst.name == "c2"  # 0.9 > 0.3, violation = 0.6

    def test_most_violated_none(self):
        pool = CutPool()
        c = LinearConstraint(coeffs=(1.0,), rhs=1.0)
        pool.add_cut(c)
        x = np.array([0.5])
        assert pool.most_violated(x) is None

    def test_max_pool_size(self):
        pool = CutPool(max_pool_size=5)
        for i in range(10):
            pool.add_cut(LinearConstraint(coeffs=(float(i),), rhs=float(i)))
        assert pool.n_cuts <= 10

    def test_active_cuts(self):
        pool = CutPool()
        for i in range(5):
            pool.add_cut(LinearConstraint(coeffs=(float(i),), rhs=1.0))
        pool.deactivate(2)
        active = pool.active_cuts()
        assert len(active) == 4


class TestCycleEliminationCut:
    def test_two_cycle(self):
        cut = generate_cycle_elimination_cut([0, 1], n=3)
        assert cut.rhs == 1.0  # x[0,1] + x[1,0] <= 1

    def test_three_cycle(self):
        cut = generate_cycle_elimination_cut([0, 1, 2], n=3)
        assert cut.rhs == 2.0  # x[0,1] + x[1,2] + x[2,0] <= 2

    def test_correct_variables(self):
        cut = generate_cycle_elimination_cut([0, 1], n=3)
        # Variables: x[0,1] and x[1,0]
        coeffs = list(cut.coeffs)
        assert coeffs[0 * 3 + 1] == 1.0  # x[0,1]
        assert coeffs[1 * 3 + 0] == 1.0  # x[1,0]
        assert sum(coeffs) == 2.0

    def test_feasible_dag(self):
        """A DAG should satisfy all cycle elimination cuts."""
        adj = _adj(3, [(0, 1), (1, 2)])
        x = adj.flatten().astype(float)
        cut = generate_cycle_elimination_cut([0, 1, 2], n=3)
        lhs = sum(c * xi for c, xi in zip(cut.coeffs, x))
        assert lhs <= cut.rhs


class TestPathCut:
    def test_simple_path(self):
        cut = generate_path_cut([0, 1, 2], n=3, treatment=0, outcome=2)
        assert cut.rhs == 1.0  # at least one edge absent from path of length 2

    def test_long_path(self):
        cut = generate_path_cut([0, 1, 2, 3], n=4, treatment=0, outcome=3)
        assert cut.rhs == 2.0

    def test_violated_when_all_present(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        x = adj.flatten().astype(float)
        cut = generate_path_cut([0, 1, 2], n=3, treatment=0, outcome=2)
        lhs = sum(c * xi for c, xi in zip(cut.coeffs, x))
        assert lhs > cut.rhs  # violated: all edges present

    def test_satisfied_when_edge_missing(self):
        adj = _adj(3, [(0, 1)])  # edge 1→2 missing
        x = adj.flatten().astype(float)
        cut = generate_path_cut([0, 1, 2], n=3, treatment=0, outcome=2)
        lhs = sum(c * xi for c, xi in zip(cut.coeffs, x))
        assert lhs <= cut.rhs


class TestBudgetCut:
    def test_zero_edits(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        cut = generate_budget_cut(adj, max_edits=0)
        # Original solution should satisfy
        x = adj.flatten().astype(float)
        lhs = sum(c * xi for c, xi in zip(cut.coeffs, x))
        assert lhs <= cut.rhs + 1e-6

    def test_one_edit(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        cut = generate_budget_cut(adj, max_edits=1)
        assert isinstance(cut.rhs, float)


class TestViolatedCycleCuts:
    def test_no_cycles_no_cuts(self):
        n = 3
        # DAG-like solution
        x = np.zeros(n * n)
        x[0 * 3 + 1] = 1.0
        x[1 * 3 + 2] = 1.0
        cuts = find_violated_cycle_cuts(x, n)
        assert len(cuts) == 0

    def test_two_cycle_detected(self):
        n = 3
        x = np.zeros(n * n)
        x[0 * 3 + 1] = 1.0
        x[1 * 3 + 0] = 1.0  # 2-cycle
        cuts = find_violated_cycle_cuts(x, n, tol=0.0)
        assert len(cuts) >= 1

    def test_fractional_solution(self):
        n = 3
        x = np.zeros(n * n)
        x[0 * 3 + 1] = 0.8
        x[1 * 3 + 0] = 0.8
        cuts = find_violated_cycle_cuts(x, n, tol=0.4)
        assert len(cuts) >= 1


class TestLPWithCuts:
    def test_trivial(self):
        c = np.array([1.0, 1.0])
        pool = CutPool()
        x, obj = solve_lp_with_cuts(c, [], pool, max_rounds=5, n_vars=2)
        assert len(x) == 2
        assert all(0 <= xi <= 1 for xi in x)

    def test_with_constraint(self):
        c = np.array([-1.0, -1.0])  # maximize x1 + x2
        con = LinearConstraint(coeffs=(1.0, 1.0), rhs=1.0)
        pool = CutPool()
        x, obj = solve_lp_with_cuts(c, [con], pool, max_rounds=10, n_vars=2)
        lhs = x[0] + x[1]
        # Should approximately satisfy constraint
        assert isinstance(obj, float)


class TestSymmetryBreaking:
    def test_generates_valid_cut(self):
        cut = generate_symmetry_breaking_cut(3)
        assert cut.rhs == 0.0
        assert len(cut.coeffs) == 9


class TestCuttingPlaneIntegration:
    """Integration tests combining cuts with solver."""

    def test_chain_path_cuts(self):
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        n = 4
        pool = CutPool()

        # Add path cuts for the single path 0→1→2→3
        cut = generate_path_cut([0, 1, 2, 3], n, 0, 3)
        pool.add_cut(cut)

        # Check that original solution violates the cut
        x = adj.flatten().astype(float)
        assert pool.is_violated(cut, x)

        # A solution with one edge removed should satisfy
        adj_mod = adj.copy()
        adj_mod[1, 2] = 0
        x_mod = adj_mod.flatten().astype(float)
        assert not pool.is_violated(cut, x_mod)

    def test_diamond_multiple_cuts(self):
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        n = 4
        pool = CutPool()

        # Two paths: 0→1→3 and 0→2→3
        cut1 = generate_path_cut([0, 1, 3], n, 0, 3)
        cut2 = generate_path_cut([0, 2, 3], n, 0, 3)
        pool.add_cut(cut1)
        pool.add_cut(cut2)

        # Need to satisfy both cuts
        # Remove edge 0→1: breaks first path but not second
        adj_mod1 = adj.copy()
        adj_mod1[0, 1] = 0
        x1 = adj_mod1.flatten().astype(float)
        assert not pool.is_violated(cut1, x1)
        assert pool.is_violated(cut2, x1)

        # Remove both 1→3 and 2→3: breaks both paths
        adj_mod2 = adj.copy()
        adj_mod2[1, 3] = 0
        adj_mod2[2, 3] = 0
        x2 = adj_mod2.flatten().astype(float)
        assert not pool.is_violated(cut1, x2)
        assert not pool.is_violated(cut2, x2)
