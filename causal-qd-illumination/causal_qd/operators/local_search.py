"""Local search operators for post-optimization of causal DAGs.

Implements greedy local search, tabu search, simulated annealing, and
hill-climbing refinement.  Each operator tries single-edge modifications
(add, remove, reverse) and selects moves that improve the score while
preserving acyclicity.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.operators.mutation import _has_cycle, _topological_sort, _can_reach
from causal_qd.scores.score_base import DecomposableScore, ScoreFunction
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class LocalSearchResult:
    """Result of a local search run."""
    dag: AdjacencyMatrix
    score: QualityScore
    iterations: int
    improvements: int
    trajectory: List[QualityScore] = field(default_factory=list)


@dataclass(frozen=True)
class EdgeOperation:
    """A single edge modification."""
    op_type: str  # "add", "remove", "reverse"
    source: int
    target: int


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LocalSearchOperator(ABC):
    """Abstract base class for local search refinement operators."""

    @abstractmethod
    def search(
        self,
        dag: AdjacencyMatrix,
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> LocalSearchResult:
        """Run local search starting from *dag*.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Initial DAG adjacency matrix.
        score_fn : ScoreFunction
            Scoring function.
        data : DataMatrix
            Observed data matrix.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        LocalSearchResult
            The optimized DAG and search statistics.
        """


# ---------------------------------------------------------------------------
# Neighbor generation
# ---------------------------------------------------------------------------


def _enumerate_neighbors(
    adj: AdjacencyMatrix,
    max_parents: int = -1,
) -> List[Tuple[EdgeOperation, AdjacencyMatrix]]:
    """Enumerate all single-edge neighbors of a DAG.

    Generates all possible add, remove, and reverse operations that
    maintain acyclicity.  Each neighbor is returned as a tuple of
    the operation and the resulting adjacency matrix.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Current DAG.
    max_parents : int
        Maximum in-degree constraint.  -1 means no limit.

    Returns
    -------
    List[Tuple[EdgeOperation, AdjacencyMatrix]]
        List of (operation, new_adjacency) pairs.
    """
    n = adj.shape[0]
    neighbors: List[Tuple[EdgeOperation, AdjacencyMatrix]] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if adj[i, j]:
                # Remove edge i -> j
                new = adj.copy()
                new[i, j] = 0
                op = EdgeOperation("remove", i, j)
                neighbors.append((op, new))

                # Reverse edge i -> j to j -> i
                new = adj.copy()
                new[i, j] = 0
                new[j, i] = 1
                if not _has_cycle(new):
                    if max_parents < 0 or new[:, i].sum() <= max_parents:
                        op = EdgeOperation("reverse", i, j)
                        neighbors.append((op, new))
            else:
                # Add edge i -> j (only if no j -> i exists)
                if not adj[j, i]:
                    new = adj.copy()
                    new[i, j] = 1
                    if max_parents < 0 or new[:, j].sum() <= max_parents:
                        if not _has_cycle(new):
                            op = EdgeOperation("add", i, j)
                            neighbors.append((op, new))

    return neighbors


def _score_diff_decomposable(
    score_fn: DecomposableScore,
    adj: AdjacencyMatrix,
    op: EdgeOperation,
    new_adj: AdjacencyMatrix,
    data: DataMatrix,
) -> float:
    """Efficiently compute score difference for decomposable scores.

    Only recomputes local scores for affected nodes.

    Parameters
    ----------
    score_fn : DecomposableScore
        Decomposable scoring function.
    adj : AdjacencyMatrix
        Current DAG.
    op : EdgeOperation
        Edge operation applied.
    new_adj : AdjacencyMatrix
        Resulting DAG.
    data : DataMatrix
        Data matrix.

    Returns
    -------
    float
        Score difference (new - old).
    """
    n = adj.shape[0]
    affected_nodes: Set[int] = set()

    if op.op_type == "add":
        affected_nodes.add(op.target)
    elif op.op_type == "remove":
        affected_nodes.add(op.target)
    elif op.op_type == "reverse":
        affected_nodes.add(op.source)
        affected_nodes.add(op.target)

    diff = 0.0
    for node in affected_nodes:
        old_parents = list(np.where(adj[:, node])[0])
        new_parents = list(np.where(new_adj[:, node])[0])
        old_local = score_fn.local_score(node, old_parents, data)
        new_local = score_fn.local_score(node, new_parents, data)
        diff += new_local - old_local

    return diff


# ---------------------------------------------------------------------------
# GreedyLocalSearch
# ---------------------------------------------------------------------------


class GreedyLocalSearch(LocalSearchOperator):
    """Greedy local search: iteratively apply the best single-edge change.

    At each step, all possible single-edge modifications (add, remove,
    reverse) are evaluated and the one that most improves the score is
    applied.  Repeats until no improvement is found or the maximum
    number of iterations is reached.

    Parameters
    ----------
    max_iterations : int
        Maximum number of greedy steps.  Default ``100``.
    max_parents : int
        Maximum in-degree for any node.  -1 means no limit.  Default ``-1``.
    min_improvement : float
        Minimum score improvement to accept a move.  Default ``1e-8``.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        max_parents: int = -1,
        min_improvement: float = 1e-8,
    ) -> None:
        self._max_iter = max_iterations
        self._max_parents = max_parents
        self._min_imp = min_improvement

    def search(
        self,
        dag: AdjacencyMatrix,
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> LocalSearchResult:
        """Run greedy hill-climbing from *dag*.

        Parameters
        ----------
        dag, score_fn, data, rng
            Standard local search parameters.

        Returns
        -------
        LocalSearchResult
        """
        current = dag.copy()
        current_score = score_fn.score(current, data)
        trajectory = [current_score]
        improvements = 0
        is_decomposable = isinstance(score_fn, DecomposableScore)

        for iteration in range(self._max_iter):
            neighbors = _enumerate_neighbors(current, self._max_parents)
            if not neighbors:
                break

            best_delta = -float("inf")
            best_adj: Optional[AdjacencyMatrix] = None

            for op, new_adj in neighbors:
                if is_decomposable:
                    delta = _score_diff_decomposable(
                        score_fn, current, op, new_adj, data  # type: ignore[arg-type]
                    )
                else:
                    new_score = score_fn.score(new_adj, data)
                    delta = new_score - current_score

                if delta > best_delta:
                    best_delta = delta
                    best_adj = new_adj

            if best_delta < self._min_imp or best_adj is None:
                break

            current = best_adj
            current_score += best_delta
            improvements += 1
            trajectory.append(current_score)

        return LocalSearchResult(
            dag=current,
            score=current_score,
            iterations=len(trajectory) - 1,
            improvements=improvements,
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# TabuSearch
# ---------------------------------------------------------------------------


class TabuSearch(LocalSearchOperator):
    """Local search with tabu list to avoid cycling.

    Maintains a list of recently applied operations and prevents
    reverting them for a configurable number of iterations.  An
    aspiration criterion allows tabu moves if they lead to the
    best-ever score.

    Parameters
    ----------
    max_iterations : int
        Maximum number of search steps.  Default ``200``.
    tabu_tenure : int
        Number of iterations an operation stays in the tabu list.
        Default ``10``.
    max_parents : int
        Maximum in-degree.  -1 means no limit.  Default ``-1``.
    aspiration : bool
        If ``True``, allow tabu moves that improve on the best-ever
        score.  Default ``True``.
    """

    def __init__(
        self,
        max_iterations: int = 200,
        tabu_tenure: int = 10,
        max_parents: int = -1,
        aspiration: bool = True,
    ) -> None:
        self._max_iter = max_iterations
        self._tenure = tabu_tenure
        self._max_parents = max_parents
        self._aspiration = aspiration

    def search(
        self,
        dag: AdjacencyMatrix,
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> LocalSearchResult:
        """Run tabu search from *dag*.

        Parameters
        ----------
        dag, score_fn, data, rng
            Standard local search parameters.

        Returns
        -------
        LocalSearchResult
        """
        current = dag.copy()
        current_score = score_fn.score(current, data)
        best_dag = current.copy()
        best_score = current_score
        trajectory = [current_score]
        improvements = 0
        is_decomposable = isinstance(score_fn, DecomposableScore)

        # Tabu list: maps EdgeOperation -> iteration when it expires
        tabu_list: Dict[EdgeOperation, int] = {}

        for iteration in range(self._max_iter):
            neighbors = _enumerate_neighbors(current, self._max_parents)
            if not neighbors:
                break

            # Evaluate all neighbors
            best_neighbor_delta = -float("inf")
            best_neighbor_adj: Optional[AdjacencyMatrix] = None
            best_neighbor_op: Optional[EdgeOperation] = None

            for op, new_adj in neighbors:
                # Check tabu status
                is_tabu = op in tabu_list and tabu_list[op] > iteration
                reverse_op = self._reverse_operation(op)
                is_reverse_tabu = (
                    reverse_op in tabu_list and tabu_list[reverse_op] > iteration
                )

                if is_decomposable:
                    delta = _score_diff_decomposable(
                        score_fn, current, op, new_adj, data  # type: ignore[arg-type]
                    )
                else:
                    new_score_val = score_fn.score(new_adj, data)
                    delta = new_score_val - current_score

                candidate_score = current_score + delta

                # Apply aspiration criterion
                aspirated = (
                    self._aspiration and candidate_score > best_score
                )

                if (is_tabu or is_reverse_tabu) and not aspirated:
                    continue

                if delta > best_neighbor_delta:
                    best_neighbor_delta = delta
                    best_neighbor_adj = new_adj
                    best_neighbor_op = op

            if best_neighbor_adj is None:
                break

            # Apply best move (even if non-improving)
            current = best_neighbor_adj
            current_score += best_neighbor_delta
            trajectory.append(current_score)

            # Update tabu list
            if best_neighbor_op is not None:
                reverse = self._reverse_operation(best_neighbor_op)
                tabu_list[reverse] = iteration + self._tenure

            if current_score > best_score:
                best_score = current_score
                best_dag = current.copy()
                improvements += 1

            # Clean expired tabu entries
            expired = [
                op for op, exp in tabu_list.items() if exp <= iteration
            ]
            for op in expired:
                del tabu_list[op]

        return LocalSearchResult(
            dag=best_dag,
            score=best_score,
            iterations=len(trajectory) - 1,
            improvements=improvements,
            trajectory=trajectory,
        )

    @staticmethod
    def _reverse_operation(op: EdgeOperation) -> EdgeOperation:
        """Return the operation that reverses *op*.

        Parameters
        ----------
        op : EdgeOperation
            Operation to reverse.

        Returns
        -------
        EdgeOperation
            The inverse operation.
        """
        if op.op_type == "add":
            return EdgeOperation("remove", op.source, op.target)
        elif op.op_type == "remove":
            return EdgeOperation("add", op.source, op.target)
        elif op.op_type == "reverse":
            return EdgeOperation("reverse", op.target, op.source)
        return op


# ---------------------------------------------------------------------------
# SimulatedAnnealing
# ---------------------------------------------------------------------------


class SimulatedAnnealing(LocalSearchOperator):
    """Simulated annealing for DAG optimization.

    Accepts worse solutions with decreasing probability according to
    a geometric cooling schedule.  At high temperatures, the search
    explores broadly; as the temperature cools, it converges to local
    optima.

    Parameters
    ----------
    max_iterations : int
        Maximum number of SA steps.  Default ``500``.
    initial_temperature : float
        Starting temperature.  Default ``10.0``.
    cooling_rate : float
        Geometric cooling factor ``T ← T * cooling_rate``.
        Default ``0.995``.
    min_temperature : float
        Minimum temperature (floor).  Default ``0.01``.
    max_parents : int
        Maximum in-degree.  Default ``-1``.
    restarts : int
        Number of random restarts from best solution so far.
        Default ``0``.
    """

    def __init__(
        self,
        max_iterations: int = 500,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 0.01,
        max_parents: int = -1,
        restarts: int = 0,
    ) -> None:
        self._max_iter = max_iterations
        self._t0 = initial_temperature
        self._cooling = cooling_rate
        self._t_min = min_temperature
        self._max_parents = max_parents
        self._restarts = restarts

    def search(
        self,
        dag: AdjacencyMatrix,
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> LocalSearchResult:
        """Run simulated annealing from *dag*.

        Parameters
        ----------
        dag, score_fn, data, rng
            Standard local search parameters.

        Returns
        -------
        LocalSearchResult
        """
        best_dag = dag.copy()
        best_score = score_fn.score(dag, data)
        all_trajectory = [best_score]
        total_improvements = 0

        for restart in range(self._restarts + 1):
            if restart == 0:
                current = dag.copy()
            else:
                current = best_dag.copy()

            current_score = score_fn.score(current, data)
            temp = self._t0
            is_decomposable = isinstance(score_fn, DecomposableScore)

            for iteration in range(self._max_iter):
                if temp < self._t_min:
                    break

                # Pick a random neighbor
                neighbors = _enumerate_neighbors(current, self._max_parents)
                if not neighbors:
                    break

                idx = rng.integers(0, len(neighbors))
                op, new_adj = neighbors[idx]

                if is_decomposable:
                    delta = _score_diff_decomposable(
                        score_fn, current, op, new_adj, data  # type: ignore[arg-type]
                    )
                else:
                    new_score = score_fn.score(new_adj, data)
                    delta = new_score - current_score

                # Metropolis criterion
                if delta > 0:
                    accept = True
                else:
                    accept_prob = math.exp(delta / max(temp, 1e-15))
                    accept = rng.random() < accept_prob

                if accept:
                    current = new_adj
                    current_score += delta
                    all_trajectory.append(current_score)

                    if current_score > best_score:
                        best_score = current_score
                        best_dag = current.copy()
                        total_improvements += 1

                temp *= self._cooling

        return LocalSearchResult(
            dag=best_dag,
            score=best_score,
            iterations=len(all_trajectory) - 1,
            improvements=total_improvements,
            trajectory=all_trajectory,
        )


# ---------------------------------------------------------------------------
# HillClimbingRefiner
# ---------------------------------------------------------------------------


class HillClimbingRefiner:
    """Refine all archive elites using local search.

    Iterates through elites in the archive and applies a local search
    operator to each, replacing the elite if the refined version has
    a higher score.

    Parameters
    ----------
    local_search : LocalSearchOperator
        The local search operator to use (greedy, tabu, SA, etc.).
    max_refine : int
        Maximum number of elites to refine per call.  Default ``10``.
    min_quality_gain : float
        Minimum improvement to accept the refined version.
        Default ``1e-6``.
    """

    def __init__(
        self,
        local_search: Optional[LocalSearchOperator] = None,
        max_refine: int = 10,
        min_quality_gain: float = 1e-6,
    ) -> None:
        self._ls = local_search or GreedyLocalSearch()
        self._max_refine = max_refine
        self._min_gain = min_quality_gain

    def refine_archive(
        self,
        elites: List[Any],
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> List[Tuple[int, LocalSearchResult]]:
        """Refine a list of archive elites.

        Parameters
        ----------
        elites : List[ArchiveEntry]
            Elites to refine.  Must have ``solution`` and ``quality``
            attributes.
        score_fn : ScoreFunction
            Scoring function.
        data : DataMatrix
            Data matrix.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        List[Tuple[int, LocalSearchResult]]
            For each refined elite, a tuple of (index, result).
            Only includes elites that were improved.
        """
        results: List[Tuple[int, LocalSearchResult]] = []
        n_to_refine = min(self._max_refine, len(elites))

        # Select elites to refine (lowest quality first for most impact)
        if len(elites) > n_to_refine:
            qualities = np.array([e.quality for e in elites])
            indices = np.argsort(qualities)[:n_to_refine]
        else:
            indices = np.arange(len(elites))

        for idx in indices:
            elite = elites[idx]
            result = self._ls.search(elite.solution, score_fn, data, rng)
            if result.score - elite.quality > self._min_gain:
                results.append((int(idx), result))

        return results

    def refine_single(
        self,
        dag: AdjacencyMatrix,
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> LocalSearchResult:
        """Apply local search to a single DAG.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Starting DAG.
        score_fn : ScoreFunction
            Scoring function.
        data : DataMatrix
            Data matrix.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        LocalSearchResult
        """
        return self._ls.search(dag, score_fn, data, rng)


# ---------------------------------------------------------------------------
# StochasticLocalSearch
# ---------------------------------------------------------------------------


class StochasticLocalSearch(LocalSearchOperator):
    """Stochastic local search: accept improving moves probabilistically.

    Instead of always choosing the best neighbor, this operator
    samples from improving neighbors with probability proportional
    to their improvement magnitude.  This helps avoid getting stuck
    in narrow valleys.

    Parameters
    ----------
    max_iterations : int
        Maximum number of steps.  Default ``200``.
    greediness : float
        Probability of choosing the best move vs. sampling.
        Default ``0.3``.
    max_parents : int
        Maximum in-degree.  Default ``-1``.
    """

    def __init__(
        self,
        max_iterations: int = 200,
        greediness: float = 0.3,
        max_parents: int = -1,
    ) -> None:
        self._max_iter = max_iterations
        self._greediness = greediness
        self._max_parents = max_parents

    def search(
        self,
        dag: AdjacencyMatrix,
        score_fn: ScoreFunction,
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> LocalSearchResult:
        """Run stochastic local search from *dag*.

        Parameters
        ----------
        dag, score_fn, data, rng
            Standard parameters.

        Returns
        -------
        LocalSearchResult
        """
        current = dag.copy()
        current_score = score_fn.score(current, data)
        best_dag = current.copy()
        best_score = current_score
        trajectory = [current_score]
        improvements = 0
        is_decomposable = isinstance(score_fn, DecomposableScore)

        for _ in range(self._max_iter):
            neighbors = _enumerate_neighbors(current, self._max_parents)
            if not neighbors:
                break

            # Evaluate all neighbors and find improving ones
            improving: List[Tuple[float, AdjacencyMatrix]] = []
            best_delta = -float("inf")
            best_adj: Optional[AdjacencyMatrix] = None

            for op, new_adj in neighbors:
                if is_decomposable:
                    delta = _score_diff_decomposable(
                        score_fn, current, op, new_adj, data  # type: ignore[arg-type]
                    )
                else:
                    ns = score_fn.score(new_adj, data)
                    delta = ns - current_score

                if delta > 0:
                    improving.append((delta, new_adj))
                if delta > best_delta:
                    best_delta = delta
                    best_adj = new_adj

            if not improving:
                break

            # Choose: greedy or stochastic
            if rng.random() < self._greediness and best_adj is not None:
                chosen = best_adj
                chosen_delta = best_delta
            else:
                # Sample proportional to improvement
                deltas = np.array([d for d, _ in improving])
                probs = deltas / deltas.sum()
                idx = rng.choice(len(improving), p=probs)
                chosen_delta, chosen = improving[idx]

            current = chosen
            current_score += chosen_delta
            trajectory.append(current_score)

            if current_score > best_score:
                best_score = current_score
                best_dag = current.copy()
                improvements += 1

        return LocalSearchResult(
            dag=best_dag,
            score=best_score,
            iterations=len(trajectory) - 1,
            improvements=improvements,
            trajectory=trajectory,
        )
