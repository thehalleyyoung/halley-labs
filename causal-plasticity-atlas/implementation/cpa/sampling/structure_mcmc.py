"""Structure MCMC with single-edge proposals.

Samples DAGs directly by proposing single-edge additions, deletions,
or reversals and accepting via the Metropolis-Hastings criterion.

This is the most straightforward MCMC approach to structure learning:
the proposal distribution modifies a single edge at a time, and the
acceptance probability is computed using the ratio of posterior scores.
Acyclicity is enforced by checking reachability after each proposal.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class EdgeProposal:
    """A proposed single-edge modification.

    Attributes
    ----------
    operation : str
        One of ``"add"``, ``"remove"``, or ``"reverse"``.
    edge : Tuple[int, int]
        ``(i, j)`` indices of the affected edge.
    score_diff : float
        Score difference induced by the proposal.
    """

    operation: str
    edge: Tuple[int, int]
    score_diff: float = 0.0


@dataclass
class StructureMCMCSample:
    """A single sample from the structure MCMC chain.

    Attributes
    ----------
    dag : NDArray
        Adjacency matrix.
    score : float
        Total log-score.
    iteration : int
        Iteration number when this sample was collected.
    """

    dag: NDArray
    score: float
    iteration: int = 0


# -------------------------------------------------------------------
# Score cache
# -------------------------------------------------------------------

class _ScoreCache:
    """Cache local score evaluations."""

    def __init__(self, score_fn: Callable[[int, Sequence[int]], float]) -> None:
        self._fn = score_fn
        self._cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        key = (node, frozenset(parents))
        if key not in self._cache:
            self._cache[key] = self._fn(node, list(parents))
        return self._cache[key]


# -------------------------------------------------------------------
# StructureMCMC
# -------------------------------------------------------------------

class StructureMCMC:
    """Structure MCMC sampler over DAG space.

    Proposes single-edge modifications (add, remove, reverse) and
    accepts them via the Metropolis-Hastings criterion.  Uses efficient
    acyclicity checking via BFS/DFS from the changed node.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    n_nodes : int
        Number of variables.
    max_parents : Optional[int]
        Upper bound on parent-set size.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        score_fn: Callable[[int, Sequence[int]], float],
        n_nodes: int,
        max_parents: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.score_fn = score_fn
        self.n_nodes = n_nodes
        self.max_parents = max_parents if max_parents is not None else min(n_nodes - 1, 5)
        self._rng = np.random.default_rng(seed)
        self._cache = _ScoreCache(score_fn)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(
        self,
        n_iterations: int,
        burnin: int = 0,
        thin: int = 1,
        restart_interval: int = 0,
    ) -> List[NDArray]:
        """Run the MCMC chain and return sampled adjacency matrices.

        Parameters
        ----------
        n_iterations : int
            Total MCMC iterations (including burn-in).
        burnin : int
            Iterations to discard.
        thin : int
            Thinning interval.
        restart_interval : int
            If > 0, restart from the best DAG seen every this many
            iterations to escape poor local modes.

        Returns
        -------
        List[NDArray]
            Sampled adjacency matrices.
        """
        current_dag = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        current_score = self._score_dag(current_dag)

        best_dag = current_dag.copy()
        best_score = current_score

        samples: List[NDArray] = []
        accepted = 0
        total = 0

        for it in range(n_iterations):
            # Random restart
            if restart_interval > 0 and it > 0 and it % restart_interval == 0:
                current_dag = best_dag.copy()
                current_score = best_score

            proposal = self.propose_edge(current_dag)
            if proposal is None:
                if it >= burnin and (it - burnin) % thin == 0:
                    samples.append(current_dag.copy())
                continue

            proposed_dag = self._apply_proposal(current_dag, proposal)
            total += 1

            # Compute score difference efficiently
            changed_nodes = self._changed_nodes(proposal)
            new_local = sum(
                self._cache.local_score(
                    n, list(np.where(proposed_dag[:, n] != 0)[0])
                )
                for n in changed_nodes
            )
            old_local = sum(
                self._cache.local_score(
                    n, list(np.where(current_dag[:, n] != 0)[0])
                )
                for n in changed_nodes
            )
            score_diff = new_local - old_local
            proposal.score_diff = score_diff

            # Neighborhood correction for MH
            old_nsize = self._neighborhood_size(current_dag)
            new_nsize = self._neighborhood_size(proposed_dag)
            log_ratio = score_diff + math.log(max(old_nsize, 1)) - math.log(max(new_nsize, 1))

            if log_ratio >= 0 or self._rng.random() < math.exp(log_ratio):
                current_dag = proposed_dag
                current_score += score_diff
                accepted += 1
                if current_score > best_score:
                    best_score = current_score
                    best_dag = current_dag.copy()

            if it >= burnin and (it - burnin) % thin == 0:
                samples.append(current_dag.copy())

        self._acceptance_rate = accepted / max(total, 1)
        self._best_dag = best_dag
        self._best_score = best_score
        return samples

    def propose_edge(self, current_dag: NDArray) -> Optional[EdgeProposal]:
        """Propose a single-edge modification to *current_dag*.

        Uniformly selects among valid add, remove, and reverse
        operations.

        Parameters
        ----------
        current_dag : NDArray
            Current adjacency matrix.

        Returns
        -------
        Optional[EdgeProposal]
            The proposed modification, or None if no valid moves exist.
        """
        proposals = self._enumerate_valid_moves(current_dag)
        if not proposals:
            return None
        idx = int(self._rng.integers(len(proposals)))
        return proposals[idx]

    def is_dag(self, adj_matrix: NDArray) -> bool:
        """Return ``True`` if *adj_matrix* encodes a valid DAG.

        Uses Kahn's algorithm for topological sorting.

        Parameters
        ----------
        adj_matrix : NDArray
            Adjacency matrix to check.

        Returns
        -------
        bool
        """
        n = adj_matrix.shape[0]
        in_degree = np.sum(adj_matrix != 0, axis=0).astype(int)
        queue = deque(i for i in range(n) if in_degree[i] == 0)
        count = 0

        while queue:
            node = queue.popleft()
            count += 1
            for child in range(n):
                if adj_matrix[node, child] != 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        return count == n

    def enumerate_proposals(
        self, current_dag: NDArray
    ) -> List[EdgeProposal]:
        """Enumerate all valid single-edge proposals.

        Parameters
        ----------
        current_dag : NDArray
            Current adjacency matrix.

        Returns
        -------
        List[EdgeProposal]
        """
        return self._enumerate_valid_moves(current_dag)

    # -----------------------------------------------------------------
    # Edge operations
    # -----------------------------------------------------------------

    def _add_edge(self, adj: NDArray, i: int, j: int) -> Optional[NDArray]:
        """Add edge i -> j if it maintains acyclicity and parent limit.

        Returns
        -------
        Optional[NDArray]
            New adjacency matrix, or None if invalid.
        """
        if adj[i, j] != 0 or i == j:
            return None
        # Check parent limit
        n_parents_j = int(np.sum(adj[:, j] != 0))
        if n_parents_j >= self.max_parents:
            return None
        new_adj = adj.copy()
        new_adj[i, j] = 1.0
        # Efficient acyclicity: check if j can reach i in the new graph
        if self._can_reach(new_adj, j, i):
            return None
        return new_adj

    def _remove_edge(self, adj: NDArray, i: int, j: int) -> Optional[NDArray]:
        """Remove edge i -> j.

        Returns
        -------
        Optional[NDArray]
            New adjacency matrix, or None if edge doesn't exist.
        """
        if adj[i, j] == 0:
            return None
        new_adj = adj.copy()
        new_adj[i, j] = 0.0
        return new_adj

    def _reverse_edge(self, adj: NDArray, i: int, j: int) -> Optional[NDArray]:
        """Reverse edge i -> j to j -> i if it maintains acyclicity.

        Returns
        -------
        Optional[NDArray]
            New adjacency matrix, or None if invalid.
        """
        if adj[i, j] == 0 or i == j:
            return None
        # Check parent limit on i
        n_parents_i = int(np.sum(adj[:, i] != 0))
        if n_parents_i >= self.max_parents:
            return None
        new_adj = adj.copy()
        new_adj[i, j] = 0.0
        new_adj[j, i] = 1.0
        # Check acyclicity: can i reach j in the new graph?
        if self._can_reach(new_adj, i, j):
            return None
        return new_adj

    # -----------------------------------------------------------------
    # Acyclicity checking
    # -----------------------------------------------------------------

    def _can_reach(self, adj: NDArray, source: int, target: int) -> bool:
        """Check if *target* is reachable from *source* via BFS.

        Parameters
        ----------
        adj : NDArray
            Adjacency matrix where ``adj[i,j] != 0`` means i -> j.
        source : int
            Starting node.
        target : int
            Node to reach.

        Returns
        -------
        bool
        """
        n = adj.shape[0]
        visited = np.zeros(n, dtype=bool)
        queue = deque([source])
        visited[source] = True

        while queue:
            node = queue.popleft()
            children = np.where(adj[node] != 0)[0]
            for child in children:
                if child == target:
                    return True
                if not visited[child]:
                    visited[child] = True
                    queue.append(child)

        return False

    # -----------------------------------------------------------------
    # Neighborhood computation
    # -----------------------------------------------------------------

    def _neighborhood_size(self, adj: NDArray) -> int:
        """Count the number of valid single-edge moves from *adj*.

        Parameters
        ----------
        adj : NDArray
            Current adjacency matrix.

        Returns
        -------
        int
        """
        return len(self._enumerate_valid_moves(adj))

    def _enumerate_valid_moves(self, adj: NDArray) -> List[EdgeProposal]:
        """Enumerate all valid single-edge proposals from *adj*.

        Returns
        -------
        List[EdgeProposal]
        """
        n = self.n_nodes
        proposals: List[EdgeProposal] = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if adj[i, j] != 0:
                    # Can remove
                    proposals.append(EdgeProposal("remove", (i, j)))
                    # Can reverse?
                    rev = self._reverse_edge(adj, i, j)
                    if rev is not None:
                        proposals.append(EdgeProposal("reverse", (i, j)))
                else:
                    # Can add?
                    added = self._add_edge(adj, i, j)
                    if added is not None:
                        proposals.append(EdgeProposal("add", (i, j)))

        return proposals

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def _score_dag(self, adj: NDArray) -> float:
        """Compute total score of a DAG.

        Parameters
        ----------
        adj : NDArray
            Adjacency matrix.

        Returns
        -------
        float
            Total log-score.
        """
        total = 0.0
        for j in range(self.n_nodes):
            parents = list(np.where(adj[:, j] != 0)[0])
            total += self._cache.local_score(j, parents)
        return total

    def _score_diff(
        self,
        adj: NDArray,
        old_adj: NDArray,
        changed_nodes: List[int],
    ) -> float:
        """Compute score difference efficiently using only changed nodes.

        Parameters
        ----------
        adj : NDArray
            New adjacency matrix.
        old_adj : NDArray
            Old adjacency matrix.
        changed_nodes : List[int]
            Nodes whose parent sets changed.

        Returns
        -------
        float
            Score difference (new - old).
        """
        diff = 0.0
        for node in changed_nodes:
            new_parents = list(np.where(adj[:, node] != 0)[0])
            old_parents = list(np.where(old_adj[:, node] != 0)[0])
            diff += (
                self._cache.local_score(node, new_parents)
                - self._cache.local_score(node, old_parents)
            )
        return diff

    def _changed_nodes(self, proposal: EdgeProposal) -> List[int]:
        """Return nodes whose parent sets are affected by *proposal*.

        Parameters
        ----------
        proposal : EdgeProposal
            The proposed modification.

        Returns
        -------
        List[int]
        """
        i, j = proposal.edge
        if proposal.operation == "add":
            return [j]
        elif proposal.operation == "remove":
            return [j]
        else:  # reverse
            return [i, j]

    def _apply_proposal(self, adj: NDArray, proposal: EdgeProposal) -> NDArray:
        """Apply an edge proposal to produce a new adjacency matrix.

        Parameters
        ----------
        adj : NDArray
            Current adjacency matrix.
        proposal : EdgeProposal
            The modification to apply.

        Returns
        -------
        NDArray
            New adjacency matrix.
        """
        i, j = proposal.edge
        if proposal.operation == "add":
            return self._add_edge(adj, i, j)
        elif proposal.operation == "remove":
            return self._remove_edge(adj, i, j)
        else:
            return self._reverse_edge(adj, i, j)

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        """Acceptance rate of the last run."""
        return getattr(self, "_acceptance_rate", 0.0)

    @staticmethod
    def edge_posterior_probabilities(samples: List[NDArray]) -> NDArray:
        """Compute empirical edge inclusion probabilities from DAG samples.

        Parameters
        ----------
        samples : List[NDArray]
            List of adjacency matrices.

        Returns
        -------
        NDArray
            Edge frequency matrix.
        """
        if not samples:
            return np.empty((0, 0))
        n = samples[0].shape[0]
        prob = np.zeros((n, n), dtype=np.float64)
        for dag in samples:
            prob += (dag != 0).astype(np.float64)
        prob /= len(samples)
        return prob

    @staticmethod
    def map_dag(samples: List[NDArray], score_fn: Callable) -> Tuple[NDArray, float]:
        """Return the MAP DAG by scoring all unique samples.

        Parameters
        ----------
        samples : List[NDArray]
            Sampled adjacency matrices.
        score_fn : Callable
            Local score function.

        Returns
        -------
        Tuple[NDArray, float]
        """
        best_dag = samples[0]
        best_score = -math.inf
        seen: set = set()

        for dag in samples:
            key = dag.tobytes()
            if key in seen:
                continue
            seen.add(key)
            n = dag.shape[0]
            score = 0.0
            for j in range(n):
                parents = list(np.where(dag[:, j] != 0)[0])
                score += score_fn(j, parents)
            if score > best_score:
                best_score = score
                best_dag = dag

        return best_dag.copy(), best_score
