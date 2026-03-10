"""Advanced heuristic solvers for DAG edit-distance and repair problems.

Implements simulated annealing, genetic algorithm, tabu search, greedy
constructive heuristic, local search with restarts, hill climbing, and
a hybrid heuristic + ILP refinement strategy.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
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

from causalcert.dag.graph import CausalDAG

NodeId = int
EditOp = Tuple[str, int, int]  # ("add"|"del"|"rev", i, j)


# ===================================================================
# Helper: DAG neighbourhood moves
# ===================================================================

def _applicable_moves(dag: CausalDAG) -> List[EditOp]:
    """Enumerate all single-edge moves that preserve acyclicity."""
    adj = dag.adj
    n = dag.n_nodes
    moves: List[EditOp] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j]:
                moves.append(("del", i, j))
                rev_adj = adj.copy()
                rev_adj[i, j] = 0
                rev_adj[j, i] = 1
                if _is_acyclic(rev_adj):
                    moves.append(("rev", i, j))
            else:
                add_adj = adj.copy()
                add_adj[i, j] = 1
                if _is_acyclic(add_adj):
                    moves.append(("add", i, j))
    return moves


def _apply_move(dag: CausalDAG, move: EditOp) -> CausalDAG:
    """Return a new DAG with *move* applied."""
    new_dag = dag.copy()
    op, i, j = move
    if op == "add":
        new_dag.add_edge(i, j)
    elif op == "del":
        new_dag.delete_edge(i, j)
    elif op == "rev":
        new_dag.delete_edge(i, j)
        new_dag.add_edge(j, i)
    return new_dag


def _is_acyclic(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for w in range(n):
            if adj[v, w]:
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
    return count == n


def _edit_distance(dag1: CausalDAG, dag2: CausalDAG) -> int:
    """Structural Hamming Distance."""
    diff = (dag1.adj != dag2.adj)
    n = dag1.n_nodes
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if diff[i, j] or diff[j, i]:
                count += 1
    return count


# ===================================================================
# Objective function protocol
# ===================================================================

@dataclass
class HeuristicResult:
    """Result of a heuristic search."""
    best_dag: CausalDAG
    best_cost: float
    n_evaluations: int
    history: List[float] = field(default_factory=list)
    edits: List[EditOp] = field(default_factory=list)


CostFunction = Callable[[CausalDAG], float]


# ===================================================================
# 1.  Simulated annealing for DAG repair
# ===================================================================

def simulated_annealing(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    max_iter: int = 5000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.01,
    rng: Optional[random.Random] = None,
) -> HeuristicResult:
    """Simulated annealing for DAG optimisation.

    At each step, proposes a random single-edge edit and accepts it with
    probability min(1, exp(-Δcost / T)).
    """
    if rng is None:
        rng = random.Random()

    current = initial_dag.copy()
    current_cost = cost_fn(current)
    best = current.copy()
    best_cost = current_cost
    temp = initial_temp
    history: List[float] = [current_cost]
    n_eval = 1

    for _ in range(max_iter):
        moves = _applicable_moves(current)
        if not moves:
            break
        move = rng.choice(moves)
        candidate = _apply_move(current, move)
        candidate_cost = cost_fn(candidate)
        n_eval += 1
        delta = candidate_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost

        history.append(best_cost)
        temp *= cooling_rate
        if temp < min_temp:
            break

    return HeuristicResult(
        best_dag=best,
        best_cost=best_cost,
        n_evaluations=n_eval,
        history=history,
    )


# ===================================================================
# 2.  Genetic algorithm for edit distance
# ===================================================================

def _crossover_dags(
    parent1: CausalDAG,
    parent2: CausalDAG,
    rng: random.Random,
) -> CausalDAG:
    """Uniform crossover: each edge independently inherited from one parent."""
    n = parent1.n_nodes
    child_adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rng.random() < 0.5:
                child_adj[i, j] = parent1.adj[i, j]
            else:
                child_adj[i, j] = parent2.adj[i, j]

    if not _is_acyclic(child_adj):
        child_adj = _repair_acyclicity(child_adj)
    return CausalDAG.from_adjacency_matrix(child_adj)


def _repair_acyclicity(adj: np.ndarray) -> np.ndarray:
    """Remove edges to break cycles via topological-order heuristic."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    order: List[int] = []
    remaining = set(range(n))

    while remaining:
        sources = [v for v in remaining if in_deg[v] == 0]
        if not sources:
            v = min(remaining, key=lambda x: in_deg[x])
            for u in range(n):
                if adj[u, v]:
                    adj[u, v] = 0
                    in_deg[v] -= 1
            sources = [v]
        for v in sources:
            if v in remaining:
                remaining.remove(v)
                order.append(v)
                for w in range(n):
                    if adj[v, w]:
                        in_deg[w] -= 1

    rank = {v: idx for idx, v in enumerate(order)}
    for i in range(n):
        for j in range(n):
            if adj[i, j] and rank.get(i, 0) >= rank.get(j, 0):
                adj[i, j] = 0
    return adj


def _mutate_dag(dag: CausalDAG, rng: random.Random, mutation_rate: float = 0.1) -> CausalDAG:
    """Mutate a DAG by randomly applying edge edits."""
    current = dag.copy()
    n = current.n_nodes
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rng.random() < mutation_rate:
                if current.adj[i, j]:
                    current.delete_edge(i, j)
                else:
                    test = current.adj.copy()
                    test[i, j] = 1
                    if _is_acyclic(test):
                        current.add_edge(i, j)
    return current


def genetic_algorithm(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    pop_size: int = 30,
    n_generations: int = 200,
    mutation_rate: float = 0.05,
    elite_fraction: float = 0.1,
    rng: Optional[random.Random] = None,
) -> HeuristicResult:
    """Genetic algorithm for DAG optimisation.

    Maintains a population of DAGs evolved via crossover, mutation,
    and elitist selection.
    """
    if rng is None:
        rng = random.Random()

    population = [initial_dag.copy()]
    for _ in range(pop_size - 1):
        population.append(_mutate_dag(initial_dag.copy(), rng, mutation_rate=0.3))

    fitness = [(cost_fn(d), d) for d in population]
    fitness.sort(key=lambda x: x[0])
    n_eval = pop_size
    best_cost, best_dag = fitness[0]
    history: List[float] = [best_cost]

    elite_count = max(1, int(pop_size * elite_fraction))

    for _gen in range(n_generations):
        new_pop: List[CausalDAG] = []
        for cost, d in fitness[:elite_count]:
            new_pop.append(d.copy())

        while len(new_pop) < pop_size:
            idx1, idx2 = rng.sample(range(min(pop_size, len(fitness))), 2)
            p1 = fitness[idx1][1]
            p2 = fitness[idx2][1]
            child = _crossover_dags(p1, p2, rng)
            child = _mutate_dag(child, rng, mutation_rate)
            new_pop.append(child)

        fitness = [(cost_fn(d), d) for d in new_pop]
        fitness.sort(key=lambda x: x[0])
        n_eval += pop_size

        if fitness[0][0] < best_cost:
            best_cost = fitness[0][0]
            best_dag = fitness[0][1].copy()
        history.append(best_cost)

    return HeuristicResult(
        best_dag=best_dag,
        best_cost=best_cost,
        n_evaluations=n_eval,
        history=history,
    )


# ===================================================================
# 3.  Tabu search with DAG-specific moves
# ===================================================================

def tabu_search(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    max_iter: int = 3000,
    tabu_tenure: int = 15,
    rng: Optional[random.Random] = None,
) -> HeuristicResult:
    """Tabu search with short-term memory and aspiration criterion.

    The tabu list forbids recently reversed moves for *tabu_tenure* steps
    unless the move leads to a new global best (aspiration).
    """
    if rng is None:
        rng = random.Random()

    current = initial_dag.copy()
    current_cost = cost_fn(current)
    best = current.copy()
    best_cost = current_cost
    tabu: Dict[EditOp, int] = {}
    history: List[float] = [current_cost]
    n_eval = 1

    for iteration in range(max_iter):
        moves = _applicable_moves(current)
        if not moves:
            break

        rng.shuffle(moves)
        best_move: Optional[EditOp] = None
        best_move_cost = float("inf")

        for move in moves:
            candidate = _apply_move(current, move)
            cand_cost = cost_fn(candidate)
            n_eval += 1

            is_tabu = move in tabu and tabu[move] > iteration
            aspiration = cand_cost < best_cost

            if (not is_tabu or aspiration) and cand_cost < best_move_cost:
                best_move = move
                best_move_cost = cand_cost

            if n_eval > 100 and best_move is not None:
                break

        if best_move is None:
            break

        current = _apply_move(current, best_move)
        current_cost = best_move_cost

        reverse_move = _reverse_of(best_move)
        tabu[reverse_move] = iteration + tabu_tenure

        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost

        history.append(best_cost)

    return HeuristicResult(
        best_dag=best,
        best_cost=best_cost,
        n_evaluations=n_eval,
        history=history,
    )


def _reverse_of(move: EditOp) -> EditOp:
    """Return the move that undoes *move*."""
    op, i, j = move
    if op == "add":
        return ("del", i, j)
    elif op == "del":
        return ("add", i, j)
    else:
        return ("rev", j, i)


# ===================================================================
# 4.  Greedy constructive heuristic
# ===================================================================

def greedy_constructive(
    target_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    max_edits: int = 50,
) -> HeuristicResult:
    """Greedily build a DAG by adding the cheapest edit at each step.

    Starts from an empty graph and adds edges from the target, choosing
    the one that most reduces cost, until no improvement is possible.
    """
    n = target_dag.n_nodes
    current = CausalDAG.empty(n)
    current_cost = cost_fn(current)
    edits: List[EditOp] = []
    history: List[float] = [current_cost]
    n_eval = 1

    for _ in range(max_edits):
        best_move: Optional[EditOp] = None
        best_improvement = 0.0

        for i in range(n):
            for j in range(n):
                if i == j or current.adj[i, j]:
                    continue
                test = current.adj.copy()
                test[i, j] = 1
                if not _is_acyclic(test):
                    continue
                candidate = _apply_move(current, ("add", i, j))
                cand_cost = cost_fn(candidate)
                n_eval += 1
                improvement = current_cost - cand_cost
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = ("add", i, j)

        if best_move is None or best_improvement <= 0:
            break

        current = _apply_move(current, best_move)
        current_cost = cost_fn(current)
        edits.append(best_move)
        history.append(current_cost)

    return HeuristicResult(
        best_dag=current,
        best_cost=current_cost,
        n_evaluations=n_eval,
        history=history,
        edits=edits,
    )


# ===================================================================
# 5.  Local search with random restarts
# ===================================================================

def local_search(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    max_iter: int = 1000,
) -> HeuristicResult:
    """Simple steepest-descent local search."""
    current = initial_dag.copy()
    current_cost = cost_fn(current)
    history: List[float] = [current_cost]
    n_eval = 1

    for _ in range(max_iter):
        moves = _applicable_moves(current)
        best_move: Optional[EditOp] = None
        best_cost_here = current_cost

        for move in moves:
            candidate = _apply_move(current, move)
            c = cost_fn(candidate)
            n_eval += 1
            if c < best_cost_here:
                best_cost_here = c
                best_move = move

        if best_move is None:
            break
        current = _apply_move(current, best_move)
        current_cost = best_cost_here
        history.append(current_cost)

    return HeuristicResult(
        best_dag=current,
        best_cost=current_cost,
        n_evaluations=n_eval,
        history=history,
    )


def local_search_with_restarts(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    n_restarts: int = 10,
    max_iter_per_restart: int = 500,
    rng: Optional[random.Random] = None,
) -> HeuristicResult:
    """Local search with random restarts.

    Each restart begins from a random perturbation of the initial DAG.
    """
    if rng is None:
        rng = random.Random()

    overall_best: Optional[HeuristicResult] = None

    for restart in range(n_restarts):
        if restart == 0:
            start = initial_dag.copy()
        else:
            start = _mutate_dag(initial_dag.copy(), rng, mutation_rate=0.2)

        result = local_search(start, cost_fn, max_iter=max_iter_per_restart)

        if overall_best is None or result.best_cost < overall_best.best_cost:
            overall_best = result

    assert overall_best is not None
    return overall_best


# ===================================================================
# 6.  Hill climbing with plateau handling
# ===================================================================

def hill_climbing_with_plateaus(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    max_iter: int = 2000,
    max_plateau_steps: int = 50,
    rng: Optional[random.Random] = None,
) -> HeuristicResult:
    """Hill climbing that allows lateral moves on plateaus.

    On a plateau (moves with Δcost = 0) the algorithm takes up to
    *max_plateau_steps* sideways moves before stopping.
    """
    if rng is None:
        rng = random.Random()

    current = initial_dag.copy()
    current_cost = cost_fn(current)
    best = current.copy()
    best_cost = current_cost
    plateau_steps = 0
    history: List[float] = [current_cost]
    n_eval = 1

    for _ in range(max_iter):
        moves = _applicable_moves(current)
        if not moves:
            break

        improving: List[Tuple[EditOp, float]] = []
        sideways: List[Tuple[EditOp, float]] = []

        rng.shuffle(moves)
        for move in moves[:min(len(moves), 200)]:
            candidate = _apply_move(current, move)
            c = cost_fn(candidate)
            n_eval += 1
            delta = c - current_cost
            if delta < -1e-10:
                improving.append((move, c))
            elif abs(delta) < 1e-10:
                sideways.append((move, c))

        if improving:
            improving.sort(key=lambda x: x[1])
            best_move, move_cost = improving[0]
            plateau_steps = 0
        elif sideways and plateau_steps < max_plateau_steps:
            best_move, move_cost = rng.choice(sideways)
            plateau_steps += 1
        else:
            break

        current = _apply_move(current, best_move)
        current_cost = move_cost
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost
        history.append(best_cost)

    return HeuristicResult(
        best_dag=best,
        best_cost=best_cost,
        n_evaluations=n_eval,
        history=history,
    )


# ===================================================================
# 7.  Hybrid: heuristic warm-start + ILP refinement
# ===================================================================

def hybrid_heuristic_ilp(
    initial_dag: CausalDAG,
    cost_fn: CostFunction,
    *,
    heuristic: str = "sa",
    heuristic_kwargs: Optional[Dict] = None,
    ilp_refine: Optional[Callable[[CausalDAG, float], HeuristicResult]] = None,
    rng: Optional[random.Random] = None,
) -> HeuristicResult:
    """Two-phase approach: heuristic search then optional ILP refinement.

    Phase 1 runs a fast heuristic to get a good incumbent solution.
    Phase 2 (if *ilp_refine* is provided) uses that solution as a warm
    start for an ILP solver to prove optimality or improve the bound.

    Parameters
    ----------
    heuristic : str
        One of ``"sa"`` (simulated annealing), ``"ga"`` (genetic algorithm),
        ``"tabu"`` (tabu search), ``"ls"`` (local search with restarts),
        ``"hc"`` (hill climbing with plateaus).
    ilp_refine : callable, optional
        ``(incumbent_dag, incumbent_cost) -> HeuristicResult``
    """
    if rng is None:
        rng = random.Random()
    kw = heuristic_kwargs or {}
    kw.setdefault("rng", rng)

    if heuristic == "sa":
        phase1 = simulated_annealing(initial_dag, cost_fn, **kw)
    elif heuristic == "ga":
        phase1 = genetic_algorithm(initial_dag, cost_fn, **kw)
    elif heuristic == "tabu":
        phase1 = tabu_search(initial_dag, cost_fn, **kw)
    elif heuristic == "ls":
        phase1 = local_search_with_restarts(initial_dag, cost_fn, **kw)
    elif heuristic == "hc":
        phase1 = hill_climbing_with_plateaus(initial_dag, cost_fn, **kw)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    if ilp_refine is not None:
        phase2 = ilp_refine(phase1.best_dag, phase1.best_cost)
        if phase2.best_cost <= phase1.best_cost:
            return phase2

    return phase1
