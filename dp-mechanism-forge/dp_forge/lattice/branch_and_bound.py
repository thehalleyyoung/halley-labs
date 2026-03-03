"""
Branch-and-bound optimization over discrete mechanism lattices.

Implements B&B search with configurable branching, bounding, node
selection, symmetry breaking, and cutting planes for tightening
LP relaxations during the search.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

from dp_forge.types import LatticePoint, QuerySpec


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SelectionStrategy(Enum):
    """Node selection strategy for B&B."""

    BEST_FIRST = auto()
    DEPTH_FIRST = auto()
    BREADTH_FIRST = auto()
    HYBRID = auto()


class BranchingHeuristic(Enum):
    """Variable selection heuristic for branching."""

    MOST_FRACTIONAL = auto()
    STRONG_BRANCHING = auto()
    PSEUDOCOST = auto()
    RELIABILITY = auto()
    FIRST_FRACTIONAL = auto()


@dataclass
class BBNode:
    """A node in the branch-and-bound tree."""

    node_id: int
    depth: int
    lower_bound: float
    upper_bound: float
    solution: Optional[npt.NDArray[np.float64]]
    parent_id: Optional[int] = None
    branching_var: Optional[int] = None
    branching_val: Optional[float] = None
    branching_dir: Optional[str] = None  # "left" or "right"
    variable_bounds_lo: Optional[npt.NDArray[np.float64]] = None
    variable_bounds_hi: Optional[npt.NDArray[np.float64]] = None
    is_pruned: bool = False
    is_integer: bool = False

    def __lt__(self, other: BBNode) -> bool:
        return self.lower_bound < other.lower_bound

    def __repr__(self) -> str:
        status = "pruned" if self.is_pruned else ("integer" if self.is_integer else "open")
        return (
            f"BBNode(id={self.node_id}, d={self.depth}, "
            f"lb={self.lower_bound:.4f}, ub={self.upper_bound:.4f}, {status})"
        )


@dataclass
class BBResult:
    """Result of branch-and-bound search."""

    optimal_solution: Optional[npt.NDArray[np.float64]]
    optimal_value: float
    lower_bound: float
    upper_bound: float
    nodes_explored: int
    nodes_pruned: int
    total_time: float
    gap: float
    is_optimal: bool
    convergence_history: List[Tuple[int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bound Computation
# ---------------------------------------------------------------------------


class BoundComputation:
    """Compute dual bounds via LP relaxation for B&B nodes.

    Solves LP relaxations of the integer program at each node
    to obtain lower bounds (for minimization) on the objective.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def compute_lp_relaxation(
        self,
        c: npt.NDArray[np.float64],
        A_ub: Optional[npt.NDArray[np.float64]],
        b_ub: Optional[npt.NDArray[np.float64]],
        A_eq: Optional[npt.NDArray[np.float64]],
        b_eq: Optional[npt.NDArray[np.float64]],
        bounds_lo: npt.NDArray[np.float64],
        bounds_hi: npt.NDArray[np.float64],
    ) -> Tuple[float, Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
        """Solve LP relaxation and return (objective, solution, duals).

        Args:
            c: Objective coefficients (minimize c·x).
            A_ub: Inequality constraint matrix.
            b_ub: Inequality RHS.
            A_eq: Equality constraint matrix.
            b_eq: Equality RHS.
            bounds_lo: Variable lower bounds.
            bounds_hi: Variable upper bounds.

        Returns:
            Tuple of (objective_value, primal_solution, dual_variables).
        """
        n = len(c)
        var_bounds = [(lo, hi) for lo, hi in zip(bounds_lo, bounds_hi)]

        try:
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=var_bounds,
                method='highs',
            )
            if result.success:
                # Extract duals if available
                duals = None
                if hasattr(result, 'ineqlin') and result.ineqlin is not None:
                    duals = getattr(result.ineqlin, 'marginals', None)
                return float(result.fun), result.x, duals
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def compute_lagrangian_bound(
        self,
        c: npt.NDArray[np.float64],
        A_ub: npt.NDArray[np.float64],
        b_ub: npt.NDArray[np.float64],
        multipliers: npt.NDArray[np.float64],
        bounds_lo: npt.NDArray[np.float64],
        bounds_hi: npt.NDArray[np.float64],
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Compute Lagrangian relaxation bound.

        L(λ) = min_x { c·x + λ·(Ax - b) } s.t. bounds
             = min_x { (c + λ·A)·x } - λ·b  s.t. bounds

        Args:
            c: Objective coefficients.
            A_ub: Constraint matrix.
            b_ub: Constraint RHS.
            multipliers: Lagrangian multipliers λ ≥ 0.
            bounds_lo: Variable lower bounds.
            bounds_hi: Variable upper bounds.

        Returns:
            Tuple of (Lagrangian bound, minimizing x).
        """
        n = len(c)
        c_lagr = c + multipliers @ A_ub
        offset = -np.dot(multipliers, b_ub)

        # Minimize c_lagr·x subject to bounds: trivial component-wise
        x_opt = np.zeros(n)
        for i in range(n):
            if c_lagr[i] >= 0:
                x_opt[i] = bounds_lo[i]
            else:
                x_opt[i] = bounds_hi[i]

        obj = np.dot(c_lagr, x_opt) + offset
        return float(obj), x_opt

    def subgradient_optimization(
        self,
        c: npt.NDArray[np.float64],
        A_ub: npt.NDArray[np.float64],
        b_ub: npt.NDArray[np.float64],
        bounds_lo: npt.NDArray[np.float64],
        bounds_hi: npt.NDArray[np.float64],
        max_iters: int = 100,
        best_ub: float = float('inf'),
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Optimize Lagrangian multipliers via subgradient method.

        Args:
            c: Objective coefficients.
            A_ub: Constraint matrix.
            b_ub: Constraint RHS.
            bounds_lo, bounds_hi: Variable bounds.
            max_iters: Maximum subgradient iterations.
            best_ub: Best known upper bound (for step size).

        Returns:
            Tuple of (best Lagrangian bound, best multipliers).
        """
        m = A_ub.shape[0]
        lam = np.zeros(m, dtype=np.float64)
        best_lb = float('-inf')
        best_lam = lam.copy()
        step_scale = 2.0
        no_improve_count = 0

        for it in range(max_iters):
            lb, x_opt = self.compute_lagrangian_bound(
                c, A_ub, b_ub, lam, bounds_lo, bounds_hi
            )

            if lb > best_lb:
                best_lb = lb
                best_lam = lam.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 10:
                    step_scale *= 0.5
                    no_improve_count = 0

            # Subgradient: g = A·x - b
            g = A_ub @ x_opt - b_ub

            g_norm_sq = np.dot(g, g)
            if g_norm_sq < 1e-12:
                break

            # Step size: Polyak's rule
            if best_ub < float('inf'):
                step = step_scale * (best_ub - lb) / g_norm_sq
            else:
                step = step_scale / (it + 1)

            lam = np.maximum(0, lam + step * g)

        return best_lb, best_lam


# ---------------------------------------------------------------------------
# Branching Strategy
# ---------------------------------------------------------------------------


class BranchingStrategy:
    """Variable and constraint selection heuristics for branching.

    Implements several heuristics for choosing which variable to
    branch on at a B&B node, and what value to branch at.
    """

    def __init__(
        self,
        heuristic: BranchingHeuristic = BranchingHeuristic.MOST_FRACTIONAL,
    ) -> None:
        self._heuristic = heuristic
        self._pseudocosts_up: Dict[int, List[float]] = {}
        self._pseudocosts_down: Dict[int, List[float]] = {}

    def select_variable(
        self,
        solution: npt.NDArray[np.float64],
        integer_vars: List[int],
        tol: float = 1e-6,
    ) -> Optional[Tuple[int, float]]:
        """Select a fractional variable to branch on.

        Args:
            solution: Current LP relaxation solution.
            integer_vars: Indices of integer-constrained variables.
            tol: Tolerance for integrality.

        Returns:
            Tuple of (variable_index, branching_value) or None if integral.
        """
        if self._heuristic == BranchingHeuristic.MOST_FRACTIONAL:
            return self._most_fractional(solution, integer_vars, tol)
        elif self._heuristic == BranchingHeuristic.FIRST_FRACTIONAL:
            return self._first_fractional(solution, integer_vars, tol)
        elif self._heuristic == BranchingHeuristic.PSEUDOCOST:
            return self._pseudocost_branch(solution, integer_vars, tol)
        else:
            return self._most_fractional(solution, integer_vars, tol)

    def _most_fractional(
        self,
        solution: npt.NDArray[np.float64],
        integer_vars: List[int],
        tol: float,
    ) -> Optional[Tuple[int, float]]:
        """Select the variable closest to 0.5 fractional part."""
        best_var = None
        best_frac = -1.0

        for idx in integer_vars:
            val = solution[idx]
            frac = val - math.floor(val)
            dist_to_half = abs(frac - 0.5)
            if frac > tol and frac < 1.0 - tol:
                if best_var is None or dist_to_half < best_frac:
                    best_var = idx
                    best_frac = dist_to_half

        if best_var is not None:
            return best_var, solution[best_var]
        return None

    def _first_fractional(
        self,
        solution: npt.NDArray[np.float64],
        integer_vars: List[int],
        tol: float,
    ) -> Optional[Tuple[int, float]]:
        """Select the first fractional variable."""
        for idx in integer_vars:
            val = solution[idx]
            frac = val - math.floor(val)
            if frac > tol and frac < 1.0 - tol:
                return idx, val
        return None

    def _pseudocost_branch(
        self,
        solution: npt.NDArray[np.float64],
        integer_vars: List[int],
        tol: float,
    ) -> Optional[Tuple[int, float]]:
        """Pseudocost branching: use historical branching scores."""
        best_var = None
        best_score = -1.0

        for idx in integer_vars:
            val = solution[idx]
            frac = val - math.floor(val)
            if frac <= tol or frac >= 1.0 - tol:
                continue

            # Compute pseudocost score
            pc_down = np.mean(self._pseudocosts_down.get(idx, [1.0]))
            pc_up = np.mean(self._pseudocosts_up.get(idx, [1.0]))
            score = max(pc_down * frac, 1e-6) * max(pc_up * (1 - frac), 1e-6)

            if score > best_score:
                best_score = score
                best_var = idx

        if best_var is not None:
            return best_var, solution[best_var]
        # Fallback to most fractional
        return self._most_fractional(solution, integer_vars, tol)

    def update_pseudocosts(
        self,
        var_idx: int,
        direction: str,
        obj_change: float,
        frac_part: float,
    ) -> None:
        """Update pseudocost data after branching."""
        if direction == "down":
            self._pseudocosts_down.setdefault(var_idx, []).append(
                obj_change / max(frac_part, 1e-10)
            )
        else:
            self._pseudocosts_up.setdefault(var_idx, []).append(
                obj_change / max(1 - frac_part, 1e-10)
            )

    def strong_branching(
        self,
        solution: npt.NDArray[np.float64],
        integer_vars: List[int],
        bound_computer: BoundComputation,
        c: npt.NDArray[np.float64],
        A_ub: Optional[npt.NDArray[np.float64]],
        b_ub: Optional[npt.NDArray[np.float64]],
        A_eq: Optional[npt.NDArray[np.float64]],
        b_eq: Optional[npt.NDArray[np.float64]],
        bounds_lo: npt.NDArray[np.float64],
        bounds_hi: npt.NDArray[np.float64],
        tol: float = 1e-6,
        max_candidates: int = 10,
    ) -> Optional[Tuple[int, float]]:
        """Strong branching: solve LP relaxations for candidate variables.

        Evaluates the LP bound improvement for branching on each
        candidate variable, selects the one with best improvement.
        """
        candidates = []
        for idx in integer_vars:
            val = solution[idx]
            frac = val - math.floor(val)
            if frac > tol and frac < 1.0 - tol:
                candidates.append((idx, val, abs(frac - 0.5)))

        # Sort by distance to 0.5, take top candidates
        candidates.sort(key=lambda x: x[2])
        candidates = candidates[:max_candidates]

        if not candidates:
            return None

        best_var = None
        best_score = float('-inf')

        for idx, val, _ in candidates:
            floor_val = math.floor(val)

            # Try branching down: x[idx] ≤ floor_val
            lo_down = bounds_lo.copy()
            hi_down = bounds_hi.copy()
            hi_down[idx] = floor_val
            lb_down, _, _ = bound_computer.compute_lp_relaxation(
                c, A_ub, b_ub, A_eq, b_eq, lo_down, hi_down
            )

            # Try branching up: x[idx] ≥ floor_val + 1
            lo_up = bounds_lo.copy()
            hi_up = bounds_hi.copy()
            lo_up[idx] = floor_val + 1
            lb_up, _, _ = bound_computer.compute_lp_relaxation(
                c, A_ub, b_ub, A_eq, b_eq, lo_up, hi_up
            )

            # Score: product of improvements (reliability branching)
            parent_lb = np.dot(c, solution)
            d_down = max(lb_down - parent_lb, 1e-10)
            d_up = max(lb_up - parent_lb, 1e-10)
            score = d_down * d_up

            if score > best_score:
                best_score = score
                best_var = (idx, val)

        return best_var


# ---------------------------------------------------------------------------
# Node Selection
# ---------------------------------------------------------------------------


class NodeSelection:
    """Node selection strategies for the B&B search tree.

    Controls which open node to explore next, trading off between
    improving bounds (best-first) and finding feasible solutions
    quickly (depth-first).
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.BEST_FIRST,
        dive_frequency: int = 10,
    ) -> None:
        self._strategy = strategy
        self._dive_frequency = dive_frequency
        self._iteration = 0
        self._heap: List[Tuple[float, int, BBNode]] = []
        self._stack: List[BBNode] = []

    def add_node(self, node: BBNode) -> None:
        """Add a node to the open set."""
        if self._strategy == SelectionStrategy.DEPTH_FIRST:
            self._stack.append(node)
        elif self._strategy == SelectionStrategy.BREADTH_FIRST:
            self._stack.insert(0, node)
        elif self._strategy == SelectionStrategy.BEST_FIRST:
            heapq.heappush(self._heap, (node.lower_bound, node.node_id, node))
        elif self._strategy == SelectionStrategy.HYBRID:
            heapq.heappush(self._heap, (node.lower_bound, node.node_id, node))
            self._stack.append(node)

    def select_node(self) -> Optional[BBNode]:
        """Select the next node to explore."""
        self._iteration += 1

        if self._strategy == SelectionStrategy.DEPTH_FIRST:
            return self._stack.pop() if self._stack else None

        elif self._strategy == SelectionStrategy.BREADTH_FIRST:
            return self._stack.pop(0) if self._stack else None

        elif self._strategy == SelectionStrategy.BEST_FIRST:
            if self._heap:
                _, _, node = heapq.heappop(self._heap)
                return node
            return None

        elif self._strategy == SelectionStrategy.HYBRID:
            # Alternate: mostly best-first with occasional diving
            if self._iteration % self._dive_frequency == 0 and self._stack:
                node = self._stack.pop()
                # Remove from heap too
                self._heap = [
                    (lb, nid, n) for lb, nid, n in self._heap
                    if nid != node.node_id
                ]
                heapq.heapify(self._heap)
                return node
            elif self._heap:
                _, _, node = heapq.heappop(self._heap)
                self._stack = [n for n in self._stack if n.node_id != node.node_id]
                return node
            elif self._stack:
                return self._stack.pop()
            return None

        return None

    @property
    def is_empty(self) -> bool:
        if self._strategy in (SelectionStrategy.DEPTH_FIRST, SelectionStrategy.BREADTH_FIRST):
            return len(self._stack) == 0
        elif self._strategy == SelectionStrategy.BEST_FIRST:
            return len(self._heap) == 0
        else:
            return len(self._heap) == 0 and len(self._stack) == 0

    @property
    def size(self) -> int:
        if self._strategy in (SelectionStrategy.DEPTH_FIRST, SelectionStrategy.BREADTH_FIRST):
            return len(self._stack)
        elif self._strategy == SelectionStrategy.BEST_FIRST:
            return len(self._heap)
        else:
            return len(self._heap)

    def prune_by_bound(self, cutoff: float) -> int:
        """Remove all nodes with lower bound ≥ cutoff."""
        count = 0
        if self._strategy in (SelectionStrategy.DEPTH_FIRST, SelectionStrategy.BREADTH_FIRST):
            before = len(self._stack)
            self._stack = [n for n in self._stack if n.lower_bound < cutoff]
            count = before - len(self._stack)
        elif self._strategy == SelectionStrategy.BEST_FIRST:
            before = len(self._heap)
            self._heap = [(lb, nid, n) for lb, nid, n in self._heap if lb < cutoff]
            heapq.heapify(self._heap)
            count = before - len(self._heap)
        else:
            before_h = len(self._heap)
            self._heap = [(lb, nid, n) for lb, nid, n in self._heap if lb < cutoff]
            heapq.heapify(self._heap)
            before_s = len(self._stack)
            self._stack = [n for n in self._stack if n.lower_bound < cutoff]
            count = (before_h - len(self._heap)) + (before_s - len(self._stack))
        return count


# ---------------------------------------------------------------------------
# Symmetry Breaking
# ---------------------------------------------------------------------------


class SymmetryBreaking:
    """Symmetry breaking for B&B via orbital fixing and isomorphism pruning.

    When the mechanism design problem has symmetries (e.g., permutations
    of outputs), symmetry breaking can dramatically reduce the search
    space by fixing orbits and pruning isomorphic subtrees.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def detect_symmetries(
        self,
        A: npt.NDArray[np.float64],
        n_vars: int,
    ) -> List[npt.NDArray[np.int64]]:
        """Detect symmetry group generators from constraint matrix structure.

        Looks for permutations π of variables that preserve the constraint
        structure: A·π(x) satisfies the same constraints as A·x.

        Args:
            A: Constraint matrix.
            n_vars: Number of variables.

        Returns:
            List of permutation generators (as arrays of indices).
        """
        generators = []
        m, n = A.shape

        # Simple column-swap symmetry detection
        col_norms = np.linalg.norm(A, axis=0)
        col_signatures = {}
        for j in range(n):
            sig = (round(col_norms[j], 6), round(float(np.sum(A[:, j])), 6))
            col_signatures.setdefault(sig, []).append(j)

        # For each group of structurally identical columns, generate swaps
        for sig, cols in col_signatures.items():
            if len(cols) < 2:
                continue
            # Adjacent transpositions within the group
            for i in range(len(cols) - 1):
                perm = np.arange(n, dtype=np.int64)
                perm[cols[i]] = cols[i + 1]
                perm[cols[i + 1]] = cols[i]
                # Verify it actually preserves constraints
                A_perm = A[:, perm]
                if self._matrices_equivalent(A, A_perm):
                    generators.append(perm)

        return generators

    def _matrices_equivalent(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
    ) -> bool:
        """Check if two matrices have the same row space (up to permutation)."""
        if A.shape != B.shape:
            return False
        # Check if rows of B are a permutation of rows of A
        A_sorted = np.sort(A, axis=0)
        B_sorted = np.sort(B, axis=0)
        return np.allclose(A_sorted, B_sorted, atol=self._tol)

    def orbital_fixing(
        self,
        solution: npt.NDArray[np.float64],
        generators: List[npt.NDArray[np.int64]],
        fixed_vars: Dict[int, float],
    ) -> Dict[int, float]:
        """Fix variables via orbital analysis.

        If variable x_j is fixed to value v, then all variables in
        the orbit of j (under the symmetry group) that haven't been
        branched on can be fixed to v.

        Args:
            solution: Current LP solution.
            generators: Symmetry group generators.
            fixed_vars: Already fixed variables {index: value}.

        Returns:
            Extended dict of fixed variables.
        """
        result = dict(fixed_vars)
        n = len(solution)

        # Compute orbits via union-find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for perm in generators:
            for i in range(n):
                if perm[i] != i:
                    union(i, int(perm[i]))

        # Group variables by orbit
        orbits: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            orbits.setdefault(root, []).append(i)

        # For each orbit, if any variable is fixed, fix all others
        for root, members in orbits.items():
            fixed_member = None
            fixed_value = None
            for m in members:
                if m in result:
                    fixed_member = m
                    fixed_value = result[m]
                    break
            if fixed_member is not None and fixed_value is not None:
                for m in members:
                    if m not in result:
                        result[m] = fixed_value

        return result

    def isomorphism_pruning(
        self,
        node: BBNode,
        explored_nodes: List[BBNode],
        generators: List[npt.NDArray[np.int64]],
    ) -> bool:
        """Check if a node is isomorphic to an already-explored node.

        Args:
            node: Node to check.
            explored_nodes: Previously explored nodes.
            generators: Symmetry group generators.

        Returns:
            True if the node should be pruned (is isomorphic).
        """
        if node.solution is None:
            return False

        for gen in generators:
            # Apply permutation to the solution
            perm_sol = node.solution[gen]
            # Check if any explored node has this permuted solution
            for exp in explored_nodes:
                if exp.solution is not None:
                    if np.allclose(perm_sol, exp.solution, atol=self._tol):
                        return True
        return False

    def lexicographic_fixing(
        self,
        n_vars: int,
        orbit_groups: List[List[int]],
    ) -> List[Tuple[int, int, str]]:
        """Generate lexicographic ordering constraints for symmetry breaking.

        For each orbit {x_{i1}, ..., x_{ik}}, add:
        x_{i1} ≤ x_{i2} ≤ ... ≤ x_{ik}

        Returns:
            List of (var_a, var_b, "<=") constraints.
        """
        constraints = []
        for group in orbit_groups:
            for i in range(len(group) - 1):
                constraints.append((group[i], group[i + 1], "<="))
        return constraints


# ---------------------------------------------------------------------------
# Cutting Planes
# ---------------------------------------------------------------------------


class CuttingPlanes:
    """Generate cutting planes to tighten LP relaxations.

    Implements Gomory mixed-integer cuts and split cuts that
    can be added to the LP relaxation at each B&B node to
    improve the lower bound.
    """

    def __init__(self, max_rounds: int = 5, tol: float = 1e-8) -> None:
        self._max_rounds = max_rounds
        self._tol = tol

    def gomory_cut(
        self,
        tableau_row: npt.NDArray[np.float64],
        rhs: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Generate a Gomory fractional cut from a simplex tableau row.

        Given tableau row: x_i + Σ a_{ij} x_j = b_i
        where b_i is fractional, the Gomory cut is:
        Σ f(a_{ij}) x_j ≥ f(b_i)
        where f(x) = x - floor(x) is the fractional part.

        Args:
            tableau_row: Coefficients from the simplex tableau.
            rhs: Right-hand side value.

        Returns:
            Tuple of (cut_coefficients, cut_rhs) for inequality cut·x ≥ rhs.
        """
        f_rhs = rhs - math.floor(rhs)
        if f_rhs < self._tol or f_rhs > 1.0 - self._tol:
            # RHS is integer, no cut needed
            return np.zeros_like(tableau_row), 0.0

        cut = np.zeros_like(tableau_row)
        for j in range(len(tableau_row)):
            f_j = tableau_row[j] - math.floor(tableau_row[j])
            if f_j <= f_rhs:
                cut[j] = f_j
            else:
                cut[j] = f_rhs * (1.0 - f_j) / (1.0 - f_rhs)

        return cut, f_rhs

    def generate_gomory_cuts(
        self,
        solution: npt.NDArray[np.float64],
        A_eq: npt.NDArray[np.float64],
        b_eq: npt.NDArray[np.float64],
        integer_vars: List[int],
        max_cuts: int = 10,
    ) -> List[Tuple[npt.NDArray[np.float64], float]]:
        """Generate multiple Gomory cuts from the current LP solution.

        Args:
            solution: Current LP solution.
            A_eq: Equality constraint matrix (from tableau).
            b_eq: Equality RHS.
            integer_vars: Indices of integer variables.
            max_cuts: Maximum number of cuts to generate.

        Returns:
            List of (cut_coefficients, cut_rhs) tuples.
        """
        cuts = []
        n = len(solution)

        for idx in integer_vars:
            val = solution[idx]
            frac = val - math.floor(val)
            if frac < self._tol or frac > 1.0 - self._tol:
                continue

            # Build a pseudo-tableau row for this variable
            row = np.zeros(n)
            row[idx] = 1.0
            # Project through equality constraints
            if A_eq is not None and len(A_eq) > 0:
                try:
                    # Solve for representation of x[idx] in terms of non-basics
                    pinv = np.linalg.pinv(A_eq)
                    rep = A_eq[0] if A_eq.shape[0] > 0 else row
                    row = rep
                except Exception:
                    pass

            cut_coeffs, cut_rhs = self.gomory_cut(row, val)
            if np.any(np.abs(cut_coeffs) > self._tol):
                cuts.append((cut_coeffs, cut_rhs))
                if len(cuts) >= max_cuts:
                    break

        return cuts

    def split_cut(
        self,
        solution: npt.NDArray[np.float64],
        pi: npt.NDArray[np.float64],
        pi0: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Generate a split cut defined by integer disjunction π·x ≤ π₀ ∨ π·x ≥ π₀+1.

        The split cut is the deepest cut that is valid for both sides
        of the disjunction.

        Args:
            solution: Current LP solution.
            pi: Split direction (integer coefficients).
            pi0: Split value (integer).

        Returns:
            Tuple of (cut_normal, cut_rhs) for cut_normal·x ≤ cut_rhs.
        """
        pi_x = np.dot(pi, solution)

        # Distance to each side of the split
        d_left = pi_x - pi0      # distance to π·x ≤ π₀
        d_right = (pi0 + 1) - pi_x  # distance to π·x ≥ π₀ + 1

        if d_left <= self._tol:
            # Already on left side, no cut
            return np.zeros_like(solution), 0.0
        if d_right <= self._tol:
            return np.zeros_like(solution), 0.0

        # Simple split cut: use the closer disjunction
        if d_left < d_right:
            return pi.copy(), pi0
        else:
            return -pi.copy(), -(pi0 + 1)

    def lift_and_project_cut(
        self,
        solution: npt.NDArray[np.float64],
        A_ub: npt.NDArray[np.float64],
        b_ub: npt.NDArray[np.float64],
        branching_var: int,
    ) -> Optional[Tuple[npt.NDArray[np.float64], float]]:
        """Generate a lift-and-project cut for a disjunctive branching.

        Args:
            solution: Current LP solution.
            A_ub: Inequality constraints.
            b_ub: Inequality RHS.
            branching_var: Variable to branch on.

        Returns:
            Cut (coefficients, rhs) or None if no cut found.
        """
        val = solution[branching_var]
        frac = val - math.floor(val)
        if frac < self._tol or frac > 1.0 - self._tol:
            return None

        n = len(solution)
        floor_val = math.floor(val)

        # Generate a simple disjunctive cut
        # x_j ≤ floor_val OR x_j ≥ floor_val + 1
        pi = np.zeros(n)
        pi[branching_var] = 1.0

        return self.split_cut(solution, pi, floor_val)


# ---------------------------------------------------------------------------
# Main Branch-and-Bound solver
# ---------------------------------------------------------------------------


class BranchAndBound:
    """Branch-and-bound solver for discrete mechanism optimization.

    Solves min c·x subject to A·x ≤ b, A_eq·x = b_eq, x ∈ Z^n
    using LP relaxation, branching, bounding, and optional
    symmetry breaking and cutting planes.
    """

    def __init__(
        self,
        selection: SelectionStrategy = SelectionStrategy.BEST_FIRST,
        branching: BranchingHeuristic = BranchingHeuristic.MOST_FRACTIONAL,
        max_nodes: int = 10000,
        max_time: float = 300.0,
        gap_tol: float = 1e-6,
        use_cuts: bool = True,
        use_symmetry: bool = False,
    ) -> None:
        self._node_selector = NodeSelection(strategy=selection)
        self._branching = BranchingStrategy(heuristic=branching)
        self._bound_computer = BoundComputation()
        self._cutting = CuttingPlanes() if use_cuts else None
        self._symmetry = SymmetryBreaking() if use_symmetry else None
        self._max_nodes = max_nodes
        self._max_time = max_time
        self._gap_tol = gap_tol
        self._next_id = 0

    def solve(
        self,
        c: npt.NDArray[np.float64],
        A_ub: Optional[npt.NDArray[np.float64]] = None,
        b_ub: Optional[npt.NDArray[np.float64]] = None,
        A_eq: Optional[npt.NDArray[np.float64]] = None,
        b_eq: Optional[npt.NDArray[np.float64]] = None,
        bounds_lo: Optional[npt.NDArray[np.float64]] = None,
        bounds_hi: Optional[npt.NDArray[np.float64]] = None,
        integer_vars: Optional[List[int]] = None,
    ) -> BBResult:
        """Solve an integer program via branch-and-bound.

        Args:
            c: Objective (minimize c·x).
            A_ub: Inequality constraints A·x ≤ b.
            b_ub: Inequality RHS.
            A_eq: Equality constraints.
            b_eq: Equality RHS.
            bounds_lo: Variable lower bounds (default 0).
            bounds_hi: Variable upper bounds (default +inf).
            integer_vars: Indices of integer variables (default: all).

        Returns:
            BBResult with optimal solution and search statistics.
        """
        n = len(c)
        c = np.array(c, dtype=np.float64)

        if bounds_lo is None:
            bounds_lo = np.zeros(n, dtype=np.float64)
        if bounds_hi is None:
            bounds_hi = np.full(n, 1e6, dtype=np.float64)
        if integer_vars is None:
            integer_vars = list(range(n))

        bounds_lo = np.array(bounds_lo, dtype=np.float64)
        bounds_hi = np.array(bounds_hi, dtype=np.float64)

        start_time = time.time()
        best_solution = None
        best_obj = float('inf')
        global_lb = float('-inf')
        nodes_explored = 0
        nodes_pruned = 0
        convergence = []

        # Solve root LP relaxation
        root_lb, root_sol, _ = self._bound_computer.compute_lp_relaxation(
            c, A_ub, b_ub, A_eq, b_eq, bounds_lo, bounds_hi
        )

        if root_sol is None:
            return BBResult(
                optimal_solution=None,
                optimal_value=float('inf'),
                lower_bound=float('inf'),
                upper_bound=float('inf'),
                nodes_explored=1,
                nodes_pruned=0,
                total_time=time.time() - start_time,
                gap=float('inf'),
                is_optimal=False,
            )

        # Check if root solution is integer
        root_is_int = self._is_integer(root_sol, integer_vars)
        if root_is_int:
            return BBResult(
                optimal_solution=root_sol,
                optimal_value=root_lb,
                lower_bound=root_lb,
                upper_bound=root_lb,
                nodes_explored=1,
                nodes_pruned=0,
                total_time=time.time() - start_time,
                gap=0.0,
                is_optimal=True,
            )

        # Create root node
        root_node = BBNode(
            node_id=self._new_id(),
            depth=0,
            lower_bound=root_lb,
            upper_bound=float('inf'),
            solution=root_sol,
            variable_bounds_lo=bounds_lo.copy(),
            variable_bounds_hi=bounds_hi.copy(),
        )
        self._node_selector.add_node(root_node)
        global_lb = root_lb

        # Optional: apply cutting planes at root
        extra_cuts_A = []
        extra_cuts_b = []
        if self._cutting is not None and A_eq is not None and b_eq is not None:
            cuts = self._cutting.generate_gomory_cuts(
                root_sol, A_eq, b_eq, integer_vars, max_cuts=5
            )
            for cut_c, cut_r in cuts:
                if np.any(np.abs(cut_c) > 1e-10):
                    # Convert ≥ cut to ≤: -cut·x ≤ -rhs
                    extra_cuts_A.append(-cut_c)
                    extra_cuts_b.append(-cut_r)

        # Build augmented constraint matrix if cuts were added
        if extra_cuts_A:
            cuts_A = np.array(extra_cuts_A)
            cuts_b = np.array(extra_cuts_b)
            if A_ub is not None:
                A_ub = np.vstack([A_ub, cuts_A])
                b_ub = np.concatenate([b_ub, cuts_b])
            else:
                A_ub = cuts_A
                b_ub = cuts_b

        # Main B&B loop
        while not self._node_selector.is_empty:
            if nodes_explored >= self._max_nodes:
                break
            if time.time() - start_time > self._max_time:
                break

            node = self._node_selector.select_node()
            if node is None:
                break

            nodes_explored += 1

            # Pruning check
            if node.lower_bound >= best_obj - self._gap_tol:
                nodes_pruned += 1
                continue

            # Solve LP relaxation at this node
            lb, sol, _ = self._bound_computer.compute_lp_relaxation(
                c, A_ub, b_ub, A_eq, b_eq,
                node.variable_bounds_lo, node.variable_bounds_hi,
            )

            if sol is None:
                # Infeasible node
                nodes_pruned += 1
                continue

            if lb >= best_obj - self._gap_tol:
                # Pruned by bound
                nodes_pruned += 1
                continue

            # Check integrality
            if self._is_integer(sol, integer_vars):
                obj = float(np.dot(c, sol))
                if obj < best_obj:
                    best_obj = obj
                    best_solution = sol.copy()
                    convergence.append((nodes_explored, best_obj))
                    # Prune nodes worse than new incumbent
                    pruned = self._node_selector.prune_by_bound(best_obj)
                    nodes_pruned += pruned
                continue

            # Branch
            branch_info = self._branching.select_variable(sol, integer_vars)
            if branch_info is None:
                continue

            var_idx, var_val = branch_info
            floor_val = math.floor(var_val)

            # Left child: x[var_idx] ≤ floor_val
            lo_left = node.variable_bounds_lo.copy()
            hi_left = node.variable_bounds_hi.copy()
            hi_left[var_idx] = floor_val
            left = BBNode(
                node_id=self._new_id(),
                depth=node.depth + 1,
                lower_bound=lb,
                upper_bound=best_obj,
                solution=None,
                parent_id=node.node_id,
                branching_var=var_idx,
                branching_val=var_val,
                branching_dir="left",
                variable_bounds_lo=lo_left,
                variable_bounds_hi=hi_left,
            )

            # Right child: x[var_idx] ≥ floor_val + 1
            lo_right = node.variable_bounds_lo.copy()
            hi_right = node.variable_bounds_hi.copy()
            lo_right[var_idx] = floor_val + 1
            right = BBNode(
                node_id=self._new_id(),
                depth=node.depth + 1,
                lower_bound=lb,
                upper_bound=best_obj,
                solution=None,
                parent_id=node.node_id,
                branching_var=var_idx,
                branching_val=var_val,
                branching_dir="right",
                variable_bounds_lo=lo_right,
                variable_bounds_hi=hi_right,
            )

            self._node_selector.add_node(left)
            self._node_selector.add_node(right)

        # Compute final gap
        if best_obj < float('inf'):
            gap = (best_obj - global_lb) / max(abs(best_obj), 1.0)
        else:
            gap = float('inf')

        return BBResult(
            optimal_solution=best_solution,
            optimal_value=best_obj,
            lower_bound=global_lb,
            upper_bound=best_obj,
            nodes_explored=nodes_explored,
            nodes_pruned=nodes_pruned,
            total_time=time.time() - start_time,
            gap=gap,
            is_optimal=gap < self._gap_tol,
            convergence_history=convergence,
        )

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _is_integer(
        self,
        solution: npt.NDArray[np.float64],
        integer_vars: List[int],
        tol: float = 1e-6,
    ) -> bool:
        """Check if solution is integer-feasible."""
        for idx in integer_vars:
            frac = solution[idx] - math.floor(solution[idx])
            if frac > tol and frac < 1.0 - tol:
                return False
        return True
