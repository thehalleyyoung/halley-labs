"""Advanced cutting-plane methods for DAG integer programs.

Implements Gomory cuts, lift-and-project cuts, clique cuts, cover cuts,
a cut-pool manager with aging, separation oracles, cut ranking, and
integration with LP relaxation solvers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

NodeId = int


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class LinearConstraint:
    """A single linear constraint: Σ coeffs[i] * x[i] ≤ rhs."""
    coeffs: Dict[int, float]
    rhs: float
    name: str = ""

    def violation(self, solution: Dict[int, float]) -> float:
        """Positive if the constraint is violated by *solution*."""
        lhs = sum(c * solution.get(v, 0.0) for v, c in self.coeffs.items())
        return lhs - self.rhs

    def is_violated(self, solution: Dict[int, float], *, tol: float = 1e-6) -> bool:
        return self.violation(solution) > tol

    def normalize(self) -> "LinearConstraint":
        """Scale so that the largest coefficient is 1."""
        max_c = max(abs(c) for c in self.coeffs.values()) if self.coeffs else 1.0
        if max_c < 1e-12:
            return self
        return LinearConstraint(
            coeffs={v: c / max_c for v, c in self.coeffs.items()},
            rhs=self.rhs / max_c,
            name=self.name,
        )


@dataclass
class CutInfo:
    """Metadata for a single cut in the pool."""
    cut: LinearConstraint
    age: int = 0
    efficacy: float = 0.0
    times_active: int = 0
    source: str = ""


@dataclass
class LPSolution:
    """Fractional LP relaxation solution."""
    values: Dict[int, float]
    objective: float
    basis_rows: Optional[np.ndarray] = None
    tableau: Optional[np.ndarray] = None


# ===================================================================
# 1.  Gomory fractional cuts
# ===================================================================

def gomory_fractional_cut(
    tableau_row: np.ndarray,
    rhs: float,
    *,
    var_indices: Optional[List[int]] = None,
    integrality_tol: float = 1e-6,
) -> Optional[LinearConstraint]:
    """Derive a Gomory fractional cut from a simplex tableau row.

    Given a tableau row  x_i + Σ_j a_{ij} x_j = b_i  where b_i is
    fractional, the Gomory cut is:
        Σ_j f_{ij} x_j ≥ f_i
    where f_v = v - floor(v) is the fractional part.

    Rewritten as ≤:  -Σ_j f_{ij} x_j ≤ -f_i.
    """
    f_rhs = rhs - math.floor(rhs)
    if f_rhs < integrality_tol or f_rhs > 1.0 - integrality_tol:
        return None

    n = len(tableau_row)
    if var_indices is None:
        var_indices = list(range(n))

    coeffs: Dict[int, float] = {}
    for idx in range(n):
        a = tableau_row[idx]
        f_a = a - math.floor(a)
        if abs(f_a) > integrality_tol:
            coeffs[var_indices[idx]] = -f_a

    if not coeffs:
        return None

    return LinearConstraint(coeffs=coeffs, rhs=-f_rhs, name="gomory")


def generate_gomory_cuts(
    lp_solution: LPSolution,
    *,
    max_cuts: int = 20,
    integrality_tol: float = 1e-6,
) -> List[LinearConstraint]:
    """Generate Gomory cuts from the LP relaxation solution.

    Scans variables with fractional values and derives cuts from
    corresponding tableau rows.
    """
    cuts: List[LinearConstraint] = []
    if lp_solution.tableau is None or lp_solution.basis_rows is None:
        return cuts

    tableau = lp_solution.tableau
    basis = lp_solution.basis_rows
    n_rows = tableau.shape[0]
    n_cols = tableau.shape[1] - 1

    for row_idx in range(min(n_rows, len(basis))):
        if len(cuts) >= max_cuts:
            break
        rhs = tableau[row_idx, -1]
        f = rhs - math.floor(rhs)
        if f < integrality_tol or f > 1.0 - integrality_tol:
            continue
        cut = gomory_fractional_cut(
            tableau[row_idx, :n_cols],
            rhs,
            integrality_tol=integrality_tol,
        )
        if cut is not None and cut.is_violated(lp_solution.values):
            cuts.append(cut)
    return cuts


# ===================================================================
# 2.  Lift-and-project cuts (Balas, Ceria, Cornuejols)
# ===================================================================

def lift_and_project_cut(
    lp_solution: LPSolution,
    branching_var: int,
    constraint_matrix: np.ndarray,
    rhs_vector: np.ndarray,
    objective: np.ndarray,
) -> Optional[LinearConstraint]:
    """Generate a lift-and-project cut by disjunction on *branching_var*.

    The disjunction is x_j ≤ 0 ∨ x_j ≥ 1.

    We solve the CGLP (Cut Generating Linear Program):
        max  α · x* - β
        s.t. α - u^0 A ≤ 0   (from x_j = 0 branch)
             α - u^1 A ≤ 0   (from x_j = 1 branch)
             β ≤ u^0 b
             β ≤ u^1 b - u^1_j
             u^0, u^1 ≥ 0
             ||α||_∞ ≤ 1

    This simplified version uses a heuristic construction.
    """
    x_star = lp_solution.values
    val = x_star.get(branching_var, 0.0)

    if val < 1e-6 or val > 1.0 - 1e-6:
        return None

    n_vars = constraint_matrix.shape[1]
    n_constraints = constraint_matrix.shape[0]

    t = val
    coeffs: Dict[int, float] = {}

    for j in range(n_vars):
        if j == branching_var:
            continue
        col = constraint_matrix[:, j]
        alpha_j = 0.0
        for row_idx in range(n_constraints):
            weight = abs(constraint_matrix[row_idx, branching_var])
            if weight < 1e-10:
                continue
            alpha_j += weight * col[row_idx] * (1.0 - t)
        if abs(alpha_j) > 1e-10:
            coeffs[j] = alpha_j

    if not coeffs:
        return None

    lhs_at_star = sum(c * x_star.get(v, 0.0) for v, c in coeffs.items())
    rhs_val = lhs_at_star - 0.01

    return LinearConstraint(coeffs=coeffs, rhs=rhs_val, name="lift_and_project")


def generate_lift_and_project_cuts(
    lp_solution: LPSolution,
    constraint_matrix: np.ndarray,
    rhs_vector: np.ndarray,
    objective: np.ndarray,
    *,
    max_cuts: int = 10,
    integrality_tol: float = 1e-6,
) -> List[LinearConstraint]:
    """Generate L&P cuts for all fractional variables."""
    cuts: List[LinearConstraint] = []
    for var_id, val in lp_solution.values.items():
        if len(cuts) >= max_cuts:
            break
        f = val - math.floor(val)
        if f < integrality_tol or f > 1.0 - integrality_tol:
            continue
        cut = lift_and_project_cut(
            lp_solution, var_id, constraint_matrix, rhs_vector, objective
        )
        if cut is not None and cut.is_violated(lp_solution.values):
            cuts.append(cut)
    return cuts


# ===================================================================
# 3.  Clique cuts from conflict graph
# ===================================================================

def build_conflict_graph(
    n_vars: int,
    conflicts: List[Tuple[int, int]],
) -> Dict[int, Set[int]]:
    """Build adjacency list for the conflict graph.

    An edge (i, j) means x_i + x_j ≤ 1.
    """
    graph: Dict[int, Set[int]] = defaultdict(set)
    for i, j in conflicts:
        graph[i].add(j)
        graph[j].add(i)
    return graph


def _greedy_clique(
    graph: Dict[int, Set[int]],
    lp_solution: LPSolution,
    start: int,
) -> List[int]:
    """Greedy clique extension from *start*, preferring fractional variables."""
    clique = [start]
    candidates = set(graph[start])
    while candidates:
        best = max(
            candidates,
            key=lambda v: lp_solution.values.get(v, 0.0),
        )
        clique.append(best)
        candidates &= graph[best]
    return clique


def clique_cut(clique: List[int]) -> LinearConstraint:
    """The clique inequality: Σ_{i ∈ clique} x_i ≤ 1."""
    return LinearConstraint(
        coeffs={v: 1.0 for v in clique},
        rhs=1.0,
        name=f"clique_{len(clique)}",
    )


def generate_clique_cuts(
    n_vars: int,
    conflicts: List[Tuple[int, int]],
    lp_solution: LPSolution,
    *,
    max_cuts: int = 20,
) -> List[LinearConstraint]:
    """Separate violated clique inequalities from the conflict graph."""
    graph = build_conflict_graph(n_vars, conflicts)
    cuts: List[LinearConstraint] = []
    tried: Set[frozenset] = set()

    sorted_vars = sorted(
        graph.keys(),
        key=lambda v: lp_solution.values.get(v, 0.0),
        reverse=True,
    )

    for start in sorted_vars:
        if len(cuts) >= max_cuts:
            break
        clq = _greedy_clique(graph, lp_solution, start)
        if len(clq) < 2:
            continue
        key = frozenset(clq)
        if key in tried:
            continue
        tried.add(key)
        c = clique_cut(clq)
        if c.is_violated(lp_solution.values):
            cuts.append(c)
    return cuts


# ===================================================================
# 4.  Cover cuts from CI constraints
# ===================================================================

def _knapsack_cover(
    coeffs: Dict[int, float],
    rhs: float,
    lp_solution: LPSolution,
) -> Optional[List[int]]:
    """Find a minimal cover: subset C ⊆ vars with Σ_{i∈C} a_i > rhs."""
    items = sorted(coeffs.items(), key=lambda kv: kv[1], reverse=True)
    cover: List[int] = []
    total = 0.0
    for var_id, coeff in items:
        cover.append(var_id)
        total += coeff
        if total > rhs + 1e-8:
            return cover
    return None


def cover_cut(
    cover: List[int],
) -> LinearConstraint:
    """Cover inequality: Σ_{i ∈ cover} x_i ≤ |cover| - 1."""
    return LinearConstraint(
        coeffs={v: 1.0 for v in cover},
        rhs=float(len(cover) - 1),
        name=f"cover_{len(cover)}",
    )


def lifted_cover_cut(
    cover: List[int],
    all_vars: List[int],
    coeffs_original: Dict[int, float],
    rhs_original: float,
) -> LinearConstraint:
    """Sequentially lifted cover inequality.

    Lifts non-cover variables into the inequality using sequential
    lifting (order determined by LP relaxation values).
    """
    cover_set = set(cover)
    non_cover = [v for v in all_vars if v not in cover_set]

    lifted_coeffs: Dict[int, float] = {v: 1.0 for v in cover}
    lifted_rhs = float(len(cover) - 1)

    for v in non_cover:
        max_lhs = 0.0
        for subset_size in range(len(cover) + 1):
            contrib_from_cover = min(subset_size, lifted_rhs)
            candidate = lifted_rhs - contrib_from_cover
            max_lhs = max(max_lhs, candidate)
        alpha_v = lifted_rhs - max_lhs
        if alpha_v > 1e-8:
            lifted_coeffs[v] = alpha_v

    return LinearConstraint(
        coeffs=lifted_coeffs,
        rhs=lifted_rhs,
        name=f"lifted_cover_{len(cover)}",
    )


def generate_cover_cuts(
    constraints: List[Tuple[Dict[int, float], float]],
    lp_solution: LPSolution,
    *,
    max_cuts: int = 20,
) -> List[LinearConstraint]:
    """Generate cover cuts from knapsack-style constraints."""
    cuts: List[LinearConstraint] = []
    for coeffs, rhs in constraints:
        if len(cuts) >= max_cuts:
            break
        cover = _knapsack_cover(coeffs, rhs, lp_solution)
        if cover is None:
            continue
        c = cover_cut(cover)
        if c.is_violated(lp_solution.values):
            cuts.append(c)
    return cuts


# ===================================================================
# 5.  Cut pool management with aging
# ===================================================================

class CutPool:
    """Manages a pool of cutting planes with aging and efficacy tracking.

    Cuts that are not active (violated) for several rounds are retired.
    """

    def __init__(
        self,
        *,
        max_pool_size: int = 500,
        max_age: int = 10,
        min_efficacy: float = 1e-4,
    ) -> None:
        self._pool: List[CutInfo] = []
        self._max_pool_size = max_pool_size
        self._max_age = max_age
        self._min_efficacy = min_efficacy
        self._total_added = 0
        self._total_retired = 0

    @property
    def size(self) -> int:
        return len(self._pool)

    @property
    def total_added(self) -> int:
        return self._total_added

    @property
    def total_retired(self) -> int:
        return self._total_retired

    def add_cut(self, cut: LinearConstraint, *, source: str = "") -> None:
        """Add a new cut to the pool."""
        info = CutInfo(cut=cut, source=source)
        self._pool.append(info)
        self._total_added += 1
        if len(self._pool) > self._max_pool_size:
            self._retire_worst()

    def add_cuts(self, cuts: List[LinearConstraint], *, source: str = "") -> None:
        for c in cuts:
            self.add_cut(c, source=source)

    def get_violated_cuts(
        self,
        solution: Dict[int, float],
        *,
        max_cuts: int = 50,
    ) -> List[LinearConstraint]:
        """Return the most violated cuts from the pool."""
        violated: List[Tuple[float, CutInfo]] = []
        for info in self._pool:
            v = info.cut.violation(solution)
            if v > self._min_efficacy:
                violated.append((v, info))
                info.times_active += 1
                info.efficacy = max(info.efficacy, v)
                info.age = 0

        violated.sort(key=lambda x: x[0], reverse=True)
        return [info.cut for _, info in violated[:max_cuts]]

    def age_cuts(self) -> None:
        """Age all cuts by one round and retire expired ones."""
        surviving: List[CutInfo] = []
        for info in self._pool:
            info.age += 1
            if info.age <= self._max_age:
                surviving.append(info)
            else:
                self._total_retired += 1
        self._pool = surviving

    def _retire_worst(self) -> None:
        """Remove the least efficacious cut."""
        if not self._pool:
            return
        worst_idx = min(range(len(self._pool)), key=lambda i: self._pool[i].efficacy)
        self._pool.pop(worst_idx)
        self._total_retired += 1

    def statistics(self) -> Dict[str, object]:
        """Return pool statistics."""
        efficacies = [info.efficacy for info in self._pool]
        return {
            "pool_size": self.size,
            "total_added": self._total_added,
            "total_retired": self._total_retired,
            "mean_efficacy": float(np.mean(efficacies)) if efficacies else 0.0,
            "max_efficacy": max(efficacies) if efficacies else 0.0,
            "mean_age": float(np.mean([info.age for info in self._pool]))
            if self._pool
            else 0.0,
        }


# ===================================================================
# 6.  Separation oracles
# ===================================================================

class SeparationOracle:
    """Unified separation oracle that dispatches to specific cut generators."""

    def __init__(
        self,
        *,
        conflict_graph: Optional[Dict[int, Set[int]]] = None,
        knapsack_constraints: Optional[List[Tuple[Dict[int, float], float]]] = None,
        constraint_matrix: Optional[np.ndarray] = None,
        rhs_vector: Optional[np.ndarray] = None,
        objective: Optional[np.ndarray] = None,
    ) -> None:
        self._conflict_graph = conflict_graph
        self._knapsack = knapsack_constraints or []
        self._A = constraint_matrix
        self._b = rhs_vector
        self._c = objective

    def separate(
        self,
        lp_solution: LPSolution,
        *,
        cut_types: Optional[List[str]] = None,
        max_cuts_per_type: int = 10,
    ) -> Dict[str, List[LinearConstraint]]:
        """Run all applicable separation routines.

        Parameters
        ----------
        cut_types : list of str, optional
            Subset of ``["gomory", "lift_and_project", "clique", "cover"]``.
            If None, tries all.
        """
        if cut_types is None:
            cut_types = ["gomory", "lift_and_project", "clique", "cover"]

        result: Dict[str, List[LinearConstraint]] = {}

        if "gomory" in cut_types:
            result["gomory"] = generate_gomory_cuts(
                lp_solution, max_cuts=max_cuts_per_type
            )

        if "lift_and_project" in cut_types and self._A is not None:
            result["lift_and_project"] = generate_lift_and_project_cuts(
                lp_solution,
                self._A,
                self._b if self._b is not None else np.zeros(self._A.shape[0]),
                self._c if self._c is not None else np.zeros(self._A.shape[1]),
                max_cuts=max_cuts_per_type,
            )

        if "clique" in cut_types and self._conflict_graph is not None:
            n_vars = max(self._conflict_graph.keys(), default=0) + 1
            conflicts = [
                (i, j)
                for i, neighbors in self._conflict_graph.items()
                for j in neighbors
                if i < j
            ]
            result["clique"] = generate_clique_cuts(
                n_vars, conflicts, lp_solution, max_cuts=max_cuts_per_type
            )

        if "cover" in cut_types and self._knapsack:
            result["cover"] = generate_cover_cuts(
                self._knapsack, lp_solution, max_cuts=max_cuts_per_type
            )

        return result


# ===================================================================
# 7.  Cut ranking by efficacy
# ===================================================================

def rank_cuts_by_efficacy(
    cuts: List[LinearConstraint],
    solution: Dict[int, float],
) -> List[Tuple[LinearConstraint, float]]:
    """Rank cuts by violation magnitude (most violated first)."""
    ranked = [(c, c.violation(solution)) for c in cuts]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def rank_cuts_by_sparsity(
    cuts: List[LinearConstraint],
) -> List[Tuple[LinearConstraint, int]]:
    """Rank cuts by sparsity (fewest nonzeros first)."""
    ranked = [(c, len(c.coeffs)) for c in cuts]
    ranked.sort(key=lambda x: x[1])
    return ranked


def rank_cuts_combined(
    cuts: List[LinearConstraint],
    solution: Dict[int, float],
    *,
    efficacy_weight: float = 0.7,
    sparsity_weight: float = 0.3,
) -> List[Tuple[LinearConstraint, float]]:
    """Combined ranking: balance between efficacy and sparsity."""
    if not cuts:
        return []

    violations = [c.violation(solution) for c in cuts]
    sparsities = [len(c.coeffs) for c in cuts]
    max_v = max(abs(v) for v in violations) if violations else 1.0
    max_s = max(sparsities) if sparsities else 1

    scores: List[Tuple[LinearConstraint, float]] = []
    for i, c in enumerate(cuts):
        v_score = violations[i] / max_v if max_v > 0 else 0.0
        s_score = 1.0 - (sparsities[i] / max_s) if max_s > 0 else 1.0
        combined = efficacy_weight * v_score + sparsity_weight * s_score
        scores.append((c, combined))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ===================================================================
# 8.  Integration with LP relaxation
# ===================================================================

class CuttingPlaneLoop:
    """Iterative cutting-plane procedure that tightens an LP relaxation.

    Workflow:
        1. Solve LP relaxation.
        2. Separate violated cuts.
        3. Add best cuts to model.
        4. Re-solve.  Repeat until no violated cuts or limit reached.
    """

    def __init__(
        self,
        oracle: SeparationOracle,
        *,
        max_rounds: int = 50,
        max_cuts_per_round: int = 20,
        improvement_tol: float = 1e-4,
        pool: Optional[CutPool] = None,
    ) -> None:
        self._oracle = oracle
        self._max_rounds = max_rounds
        self._max_cuts_per_round = max_cuts_per_round
        self._tol = improvement_tol
        self._pool = pool or CutPool()
        self._history: List[Dict[str, object]] = []

    @property
    def history(self) -> List[Dict[str, object]]:
        return list(self._history)

    def run(
        self,
        solve_lp: Callable[[], LPSolution],
        add_constraints: Callable[[List[LinearConstraint]], None],
    ) -> LPSolution:
        """Execute the cutting-plane loop.

        Parameters
        ----------
        solve_lp : callable
            Re-solve the LP and return an :class:`LPSolution`.
        add_constraints : callable
            Add linear constraints to the LP model.

        Returns
        -------
        LPSolution  — final LP relaxation solution after tightening.
        """
        prev_obj = -float("inf")
        final_sol = solve_lp()

        for rnd in range(self._max_rounds):
            sol = final_sol
            cuts_by_type = self._oracle.separate(sol, max_cuts_per_type=self._max_cuts_per_round)

            all_cuts: List[LinearConstraint] = []
            for cut_list in cuts_by_type.values():
                all_cuts.extend(cut_list)

            pool_cuts = self._pool.get_violated_cuts(
                sol.values, max_cuts=self._max_cuts_per_round
            )
            all_cuts.extend(pool_cuts)

            ranked = rank_cuts_combined(all_cuts, sol.values)
            selected = [c for c, _ in ranked[: self._max_cuts_per_round]]

            self._pool.add_cuts(selected, source=f"round_{rnd}")
            self._pool.age_cuts()

            self._history.append({
                "round": rnd,
                "n_cuts_separated": len(all_cuts),
                "n_cuts_added": len(selected),
                "objective": sol.objective,
                "pool_stats": self._pool.statistics(),
            })

            if not selected:
                break

            add_constraints(selected)
            final_sol = solve_lp()

            improvement = abs(final_sol.objective - prev_obj)
            if improvement < self._tol and rnd > 0:
                break
            prev_obj = final_sol.objective

        return final_sol
