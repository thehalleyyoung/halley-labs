"""MCMC samplers over topological orderings and DAG partitions.

Implements Order MCMC (Friedman & Koller 2003) which samples topological
orderings via Metropolis-Hastings, then enumerates DAG-consistent parent
sets.  Also provides Partition MCMC (Kuipers & Moffa 2017) which samples
over ordered partitions of nodes for improved mixing.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import (
    AdjacencyMatrix,
    DataMatrix,
    QualityScore,
    TopologicalOrder,
)
from causal_qd.scores.score_base import DecomposableScore
from causal_qd.operators.mutation import _has_cycle, _topological_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _order_to_dag_best_parents(
    order: TopologicalOrder,
    score_fn: DecomposableScore,
    data: DataMatrix,
    max_parents: int,
) -> AdjacencyMatrix:
    """Build a DAG from *order* by greedily choosing the best parent set
    for each node among predecessors in the ordering."""
    n = len(order)
    adj = np.zeros((n, n), dtype=np.int8)
    pos = {node: i for i, node in enumerate(order)}

    for idx, node in enumerate(order):
        candidates = order[:idx]
        if not candidates:
            continue
        best_parents: List[int] = []
        best_score = score_fn.local_score(node, [], data)

        # Greedy forward selection up to max_parents
        available = list(candidates)
        for _ in range(min(max_parents, len(available))):
            best_add: Optional[int] = None
            best_new = best_score
            for c in available:
                trial = sorted(best_parents + [c])
                s = score_fn.local_score(node, trial, data)
                if s > best_new:
                    best_new = s
                    best_add = c
            if best_add is None:
                break
            best_parents.append(best_add)
            best_score = best_new
            available.remove(best_add)

        for p in best_parents:
            adj[p, node] = 1
    return adj


def _order_score(
    order: TopologicalOrder,
    score_fn: DecomposableScore,
    data: DataMatrix,
    max_parents: int,
) -> float:
    """Compute the log-marginal-likelihood of *order* by summing the best
    local scores over admissible parent sets."""
    n = len(order)
    total = 0.0
    for idx, node in enumerate(order):
        candidates = order[:idx]
        best_parents: List[int] = []
        best_score = score_fn.local_score(node, [], data)
        available = list(candidates)
        for _ in range(min(max_parents, len(available))):
            best_add: Optional[int] = None
            best_new = best_score
            for c in available:
                trial = sorted(best_parents + [c])
                s = score_fn.local_score(node, trial, data)
                if s > best_new:
                    best_new = s
                    best_add = c
            if best_add is None:
                break
            best_parents.append(best_add)
            best_score = best_new
            available.remove(best_add)
        total += best_score
    return total


def _order_to_dag_exact_parents(
    order: TopologicalOrder,
    score_fn: DecomposableScore,
    data: DataMatrix,
    max_parents: int,
) -> AdjacencyMatrix:
    """Build optimal DAG by exact enumeration of parent sets.

    For each node, enumerate all subsets of predecessors (in the ordering)
    up to size max_parents and pick the highest-scoring one. This is exact
    but exponential in max_parents, so use only for max_parents <= 4.
    """
    from itertools import combinations

    n = len(order)
    adj = np.zeros((n, n), dtype=np.int8)

    for idx, node in enumerate(order):
        predecessors = order[:idx]
        best_score = score_fn.local_score(node, [], data)
        best_parents: List[int] = []

        # Enumerate all subsets of predecessors up to size max_parents
        for k in range(1, min(max_parents, len(predecessors)) + 1):
            for parents in combinations(predecessors, k):
                parent_list = sorted(parents)
                score = score_fn.local_score(node, parent_list, data)
                if score > best_score:
                    best_score = score
                    best_parents = parent_list

        for p in best_parents:
            adj[p, node] = 1

    return adj


def _order_score_exact(
    order: TopologicalOrder,
    score_fn: DecomposableScore,
    data: DataMatrix,
    max_parents: int,
) -> float:
    """Compute the log-marginal-likelihood of *order* by exact enumeration
    of all parent subsets up to size max_parents."""
    from itertools import combinations

    total = 0.0
    for idx, node in enumerate(order):
        predecessors = order[:idx]
        best_score = score_fn.local_score(node, [], data)

        for k in range(1, min(max_parents, len(predecessors)) + 1):
            for parents in combinations(predecessors, k):
                parent_list = sorted(parents)
                score = score_fn.local_score(node, parent_list, data)
                if score > best_score:
                    best_score = score
        total += best_score
    return total


def _order_to_dag(
    order: TopologicalOrder,
    score_fn: DecomposableScore,
    data: DataMatrix,
    max_parents: int,
) -> AdjacencyMatrix:
    """Build optimal DAG from an ordering, using exact enumeration when feasible."""
    if max_parents <= 4:
        return _order_to_dag_exact_parents(order, score_fn, data, max_parents)
    return _order_to_dag_best_parents(order, score_fn, data, max_parents)


def _order_score_auto(
    order: TopologicalOrder,
    score_fn: DecomposableScore,
    data: DataMatrix,
    max_parents: int,
) -> float:
    """Score an ordering, using exact enumeration when feasible."""
    if max_parents <= 4:
        return _order_score_exact(order, score_fn, data, max_parents)
    return _order_score(order, score_fn, data, max_parents)


def _swap_adjacent(order: TopologicalOrder, rng: np.random.Generator) -> TopologicalOrder:
    """Return a new ordering with two adjacent elements swapped."""
    n = len(order)
    if n < 2:
        return list(order)
    i = rng.integers(0, n - 1)
    new = list(order)
    new[i], new[i + 1] = new[i + 1], new[i]
    return new


def _swap_random(order: TopologicalOrder, rng: np.random.Generator) -> TopologicalOrder:
    """Return a new ordering with two random elements swapped."""
    n = len(order)
    if n < 2:
        return list(order)
    i, j = rng.choice(n, size=2, replace=False)
    new = list(order)
    new[i], new[j] = new[j], new[i]
    return new


# ---------------------------------------------------------------------------
# Order MCMC
# ---------------------------------------------------------------------------

@dataclass
class OrderMCMCResult:
    """Result container for an Order MCMC run."""
    sampled_dags: List[AdjacencyMatrix]
    sampled_orders: List[TopologicalOrder]
    scores: List[float]
    acceptance_rate: float
    best_dag: AdjacencyMatrix
    best_score: float
    trace: List[float] = field(default_factory=list)
    r_hat: Optional[float] = None
    ess: Optional[float] = None


class OrderMCMC:
    """MCMC sampler over topological orderings (Friedman & Koller 2003).

    The state space is the set of permutations of node indices.  At each step
    we propose a neighbouring ordering (adjacent-swap or random-swap) and
    accept/reject via the Metropolis-Hastings ratio computed from the
    order-marginal likelihood.

    Parameters
    ----------
    score_fn : DecomposableScore
        Decomposable scoring function (BIC, BDeu, BGe, …).
    max_parents : int
        Maximum number of parents per node.
    proposal : str
        ``"adjacent"`` or ``"random"`` swap proposal.
    """

    def __init__(
        self,
        score_fn: DecomposableScore,
        max_parents: int = 5,
        proposal: str = "adjacent",
    ) -> None:
        self.score_fn = score_fn
        self.max_parents = max_parents
        if proposal not in ("adjacent", "random"):
            raise ValueError(f"Unknown proposal type: {proposal}")
        self._propose = _swap_adjacent if proposal == "adjacent" else _swap_random

    # ------------------------------------------------------------------ run
    def run(
        self,
        data: DataMatrix,
        n_samples: int = 1000,
        burn_in: int = 500,
        thinning: int = 1,
        initial_order: Optional[TopologicalOrder] = None,
        rng: Optional[np.random.Generator] = None,
        compute_diagnostics: bool = False,
    ) -> OrderMCMCResult:
        """Run the Order MCMC sampler.

        Parameters
        ----------
        data : DataMatrix
            N × p data matrix.
        n_samples : int
            Number of post-burn-in samples to collect.
        burn_in : int
            Number of initial iterations to discard.
        thinning : int
            Keep every *thinning*-th sample after burn-in.
        initial_order : optional
            Starting topological order.  If ``None``, a random permutation
            is used.
        rng : optional
            NumPy random generator.
        compute_diagnostics : bool
            If True, compute R̂ (Gelman-Rubin) and ESS on the score trace
            and store them in the result.  R̂ is computed by splitting the
            trace into two halves.

        Returns
        -------
        OrderMCMCResult
        """
        rng = rng or np.random.default_rng()
        p = data.shape[1]

        # Initialise
        current_order: TopologicalOrder = (
            list(initial_order) if initial_order is not None
            else list(rng.permutation(p))
        )
        current_score = _order_score_auto(current_order, self.score_fn, data, self.max_parents)

        total_iters = burn_in + n_samples * thinning
        sampled_orders: List[TopologicalOrder] = []
        sampled_dags: List[AdjacencyMatrix] = []
        scores: List[float] = []
        trace: List[float] = []
        accepted = 0

        best_order = list(current_order)
        best_score = current_score

        for it in range(total_iters):
            # Propose
            proposed = self._propose(current_order, rng)
            proposed_score = _order_score_auto(proposed, self.score_fn, data, self.max_parents)

            # Accept / reject (Metropolis)
            log_alpha = proposed_score - current_score
            if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                current_order = proposed
                current_score = proposed_score
                accepted += 1

            trace.append(current_score)

            if current_score > best_score:
                best_score = current_score
                best_order = list(current_order)

            # Collect sample
            if it >= burn_in and (it - burn_in) % thinning == 0:
                sampled_orders.append(list(current_order))
                dag = _order_to_dag(
                    current_order, self.score_fn, data, self.max_parents,
                )
                sampled_dags.append(dag)
                scores.append(current_score)

        best_dag = _order_to_dag(
            best_order, self.score_fn, data, self.max_parents,
        )

        r_hat_val: Optional[float] = None
        ess_val: Optional[float] = None
        if compute_diagnostics and len(trace) >= 4:
            from causal_qd.sampling.convergence import GelmanRubin, EffectiveSampleSize
            half = len(trace) // 2
            chains = [trace[:half], trace[half:2 * half]]
            try:
                gr = GelmanRubin()
                r_hat_val = gr.compute(chains).r_hat
            except ValueError:
                pass
            ess_calc = EffectiveSampleSize()
            ess_val = ess_calc.compute(trace).ess

        return OrderMCMCResult(
            sampled_dags=sampled_dags,
            sampled_orders=sampled_orders,
            scores=scores,
            acceptance_rate=accepted / max(total_iters, 1),
            best_dag=best_dag,
            best_score=best_score,
            trace=trace,
            r_hat=r_hat_val,
            ess=ess_val,
        )

    # ------------------------------------------------------------- parallel
    def run_parallel_chains(
        self,
        data: DataMatrix,
        n_chains: int = 4,
        n_samples: int = 1000,
        burn_in: int = 500,
        thinning: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List[OrderMCMCResult]:
        """Run multiple independent chains for convergence diagnostics."""
        rng = rng or np.random.default_rng()
        results: List[OrderMCMCResult] = []
        for c in range(n_chains):
            chain_rng = np.random.default_rng(rng.integers(0, 2**32))
            res = self.run(
                data,
                n_samples=n_samples,
                burn_in=burn_in,
                thinning=thinning,
                rng=chain_rng,
            )
            results.append(res)
        return results

    # ------------------------------------------------- posterior edge probs
    def edge_probabilities(self, result: OrderMCMCResult) -> npt.NDArray[np.float64]:
        """Compute posterior edge inclusion probabilities from sampled DAGs."""
        if not result.sampled_dags:
            raise ValueError("No sampled DAGs in result.")
        n = result.sampled_dags[0].shape[0]
        counts = np.zeros((n, n), dtype=np.float64)
        for dag in result.sampled_dags:
            counts += dag.astype(np.float64)
        return counts / len(result.sampled_dags)


# ---------------------------------------------------------------------------
# Partition MCMC
# ---------------------------------------------------------------------------

@dataclass
class PartitionMCMCResult:
    """Result container for a Partition MCMC run."""
    sampled_dags: List[AdjacencyMatrix]
    sampled_partitions: List[List[List[int]]]
    scores: List[float]
    acceptance_rate: float
    best_dag: AdjacencyMatrix
    best_score: float
    trace: List[float] = field(default_factory=list)


def _partition_to_order(partition: List[List[int]]) -> TopologicalOrder:
    """Convert an ordered partition to a topological order by sorting
    within each block (deterministic for MH detailed balance)."""
    order: TopologicalOrder = []
    for block in partition:
        shuffled = list(block)
        shuffled.sort()
        order.extend(shuffled)
    return order


def _propose_partition(
    partition: List[List[int]],
    rng: np.random.Generator,
) -> List[List[int]]:
    """Propose a new partition by one of three moves:
    1. Swap two nodes between adjacent blocks
    2. Split a block in two
    3. Merge two adjacent blocks
    """
    partition = [list(b) for b in partition]
    n_blocks = len(partition)

    if n_blocks <= 1:
        # Only split is possible
        if len(partition[0]) <= 1:
            return partition
        move = "split"
    else:
        move = rng.choice(["swap", "split", "merge"])

    if move == "swap" and n_blocks >= 2:
        idx = rng.integers(0, n_blocks - 1)
        b1, b2 = partition[idx], partition[idx + 1]
        if b1 and b2:
            i = rng.integers(0, len(b1))
            j = rng.integers(0, len(b2))
            b1[i], b2[j] = b2[j], b1[i]
            partition[idx] = b1
            partition[idx + 1] = b2

    elif move == "split":
        # Pick a block with > 1 element and split
        splittable = [i for i, b in enumerate(partition) if len(b) > 1]
        if splittable:
            idx = rng.choice(splittable)
            block = list(partition[idx])
            rng.shuffle(block)
            mid = rng.integers(1, len(block))
            partition[idx] = block[:mid]
            partition.insert(idx + 1, block[mid:])

    elif move == "merge" and n_blocks >= 2:
        idx = rng.integers(0, n_blocks - 1)
        merged = partition[idx] + partition[idx + 1]
        partition[idx] = merged
        del partition[idx + 1]

    # Remove empty blocks
    partition = [b for b in partition if b]
    return partition


class PartitionMCMC:
    """MCMC sampler over ordered partitions of nodes (Kuipers & Moffa 2017).

    Ordered partitions generalise topological orderings by grouping nodes
    into equivalence blocks.  This yields a larger state space with better
    mixing properties compared to :class:`OrderMCMC`.

    Parameters
    ----------
    score_fn : DecomposableScore
        Decomposable scoring function.
    max_parents : int
        Maximum number of parents per node.
    """

    def __init__(
        self,
        score_fn: DecomposableScore,
        max_parents: int = 5,
    ) -> None:
        self.score_fn = score_fn
        self.max_parents = max_parents

    def _partition_score(
        self,
        partition: List[List[int]],
        data: DataMatrix,
        rng: np.random.Generator,
    ) -> Tuple[float, AdjacencyMatrix]:
        """Score a partition by converting to order and scoring."""
        order = _partition_to_order(partition)
        score = _order_score_auto(order, self.score_fn, data, self.max_parents)
        dag = _order_to_dag(order, self.score_fn, data, self.max_parents)
        return score, dag

    def run(
        self,
        data: DataMatrix,
        n_samples: int = 1000,
        burn_in: int = 500,
        thinning: int = 1,
        initial_partition: Optional[List[List[int]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> PartitionMCMCResult:
        """Run the Partition MCMC sampler.

        Parameters
        ----------
        data : DataMatrix
            N × p data matrix.
        n_samples : int
            Number of post-burn-in samples to collect.
        burn_in : int
            Number of initial iterations to discard.
        thinning : int
            Keep every *thinning*-th sample.
        initial_partition : optional
            Starting ordered partition.  Default: single block with all nodes.
        rng : optional
            NumPy random generator.

        Returns
        -------
        PartitionMCMCResult
        """
        rng = rng or np.random.default_rng()
        p = data.shape[1]

        # Initialise
        if initial_partition is not None:
            current_partition = [list(b) for b in initial_partition]
        else:
            nodes = list(rng.permutation(p))
            current_partition = [nodes]

        current_score, current_dag = self._partition_score(current_partition, data, rng)

        total_iters = burn_in + n_samples * thinning
        sampled_partitions: List[List[List[int]]] = []
        sampled_dags: List[AdjacencyMatrix] = []
        scores: List[float] = []
        trace: List[float] = []
        accepted = 0

        best_partition = [list(b) for b in current_partition]
        best_score = current_score
        best_dag = current_dag.copy()

        for it in range(total_iters):
            # Propose
            proposed_partition = _propose_partition(current_partition, rng)
            proposed_score, proposed_dag = self._partition_score(
                proposed_partition, data, rng,
            )

            # Accept / reject (Metropolis)
            log_alpha = proposed_score - current_score
            if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                current_partition = proposed_partition
                current_score = proposed_score
                current_dag = proposed_dag
                accepted += 1

            trace.append(current_score)

            if current_score > best_score:
                best_score = current_score
                best_partition = [list(b) for b in current_partition]
                best_dag = current_dag.copy()

            if it >= burn_in and (it - burn_in) % thinning == 0:
                sampled_partitions.append([list(b) for b in current_partition])
                sampled_dags.append(current_dag.copy())
                scores.append(current_score)

        return PartitionMCMCResult(
            sampled_dags=sampled_dags,
            sampled_partitions=sampled_partitions,
            scores=scores,
            acceptance_rate=accepted / max(total_iters, 1),
            best_dag=best_dag,
            best_score=best_score,
            trace=trace,
        )

    def edge_probabilities(self, result: PartitionMCMCResult) -> npt.NDArray[np.float64]:
        """Compute posterior edge inclusion probabilities."""
        if not result.sampled_dags:
            raise ValueError("No sampled DAGs in result.")
        n = result.sampled_dags[0].shape[0]
        counts = np.zeros((n, n), dtype=np.float64)
        for dag in result.sampled_dags:
            counts += dag.astype(np.float64)
        return counts / len(result.sampled_dags)
