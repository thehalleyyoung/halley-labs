"""Partition MCMC for improved mixing over DAG space.

Generalises order MCMC by sampling over partitions (bucketed orderings)
of nodes, allowing more radical proposals via block splits and merges.

A partition groups nodes into ordered blocks.  Within a block, nodes
are topologically unordered relative to each other, meaning any
ordering of nodes within a block is equally valid.  This coarsening of
the order space allows larger jumps in the MCMC chain, improving
mixing for multi-modal posteriors (Kuipers & Moffa, 2017).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class Partition:
    """A partition of node indices into ordered blocks.

    Attributes
    ----------
    blocks : List[List[int]]
        Ordered list of blocks; nodes within a block are unordered
        relative to each other.
    score : float
        Log-score of the partition.
    """

    blocks: List[List[int]] = field(default_factory=list)
    score: float = 0.0

    def copy(self) -> "Partition":
        """Return a deep copy of this partition."""
        return Partition(
            blocks=[list(b) for b in self.blocks],
            score=self.score,
        )

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def predecessors(self, node: int) -> List[int]:
        """Return all nodes in blocks strictly before the block containing *node*."""
        preds: List[int] = []
        for block in self.blocks:
            if node in block:
                break
            preds.extend(block)
        return preds

    def block_predecessors(self, block_idx: int) -> List[int]:
        """Return all nodes in blocks before *block_idx*."""
        preds: List[int] = []
        for i in range(block_idx):
            preds.extend(self.blocks[i])
        return preds

    def all_nodes(self) -> List[int]:
        """Return flat list of all nodes."""
        return [n for b in self.blocks for n in b]


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
# PartitionMCMC
# -------------------------------------------------------------------

class PartitionMCMC:
    """Partition MCMC sampler.

    Explores the posterior over DAGs by sampling partitions of nodes
    into ordered blocks.  Each partition defines an equivalence class
    of linear orderings.  Proposals include splitting a block in two,
    merging adjacent blocks, and swapping elements between adjacent
    blocks.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    n_nodes : int
        Number of variables.
    max_parents : Optional[int]
        Upper bound on the parent-set size.
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
        self._best_parent_cache: Dict[Tuple[int, FrozenSet[int]], Tuple[float, List[int]]] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(
        self,
        n_iterations: int,
        burnin: int = 0,
        thin: int = 1,
    ) -> List[Partition]:
        """Run the MCMC chain and return collected partition samples.

        Parameters
        ----------
        n_iterations : int
            Total number of MCMC iterations (including burn-in).
        burnin : int
            Number of initial iterations to discard.
        thin : int
            Keep every *thin*-th sample after burn-in.

        Returns
        -------
        List[Partition]
            Collected partition samples.
        """
        current = self._initial_partition()
        current.score = self._score_partition(current)

        samples: List[Partition] = []
        self._accepted = 0
        self._total = 0

        for it in range(n_iterations):
            proposed, proposal_ratio = self.propose_partition(current)
            proposed.score = self._score_partition(proposed)
            self._total += 1

            alpha = self._acceptance_probability(
                proposed.score, current.score, proposal_ratio
            )
            if self._rng.random() < alpha:
                current = proposed
                self._accepted += 1

            if it >= burnin and (it - burnin) % thin == 0:
                samples.append(current.copy())

        return samples

    def propose_partition(
        self, current: Partition
    ) -> Tuple[Partition, float]:
        """Propose a new partition via a split, merge, or swap move.

        Parameters
        ----------
        current : Partition
            Current partition state.

        Returns
        -------
        Tuple[Partition, float]
            ``(proposed_partition, proposal_ratio)`` where proposal_ratio
            is ``q(current|proposed) / q(proposed|current)`` for the
            Metropolis-Hastings correction.
        """
        # Determine which moves are available
        can_split = any(len(b) > 1 for b in current.blocks)
        can_merge = current.n_blocks > 1
        can_swap = current.n_blocks > 1

        moves = []
        if can_split:
            moves.append("split")
        if can_merge:
            moves.append("merge")
        if can_swap:
            moves.append("swap")

        if not moves:
            return current.copy(), 1.0

        move = moves[int(self._rng.integers(len(moves)))]
        n_forward_moves = len(moves)

        if move == "split":
            proposed, move_ratio = self._split_move(current)
        elif move == "merge":
            proposed, move_ratio = self._merge_move(current)
        else:
            proposed, move_ratio = self._swap_move(current)

        # Count reverse moves available
        can_split_rev = any(len(b) > 1 for b in proposed.blocks)
        can_merge_rev = proposed.n_blocks > 1
        can_swap_rev = proposed.n_blocks > 1
        n_reverse_moves = sum([can_split_rev, can_merge_rev, can_swap_rev])

        proposal_ratio = move_ratio * (n_forward_moves / max(n_reverse_moves, 1))
        return proposed, proposal_ratio

    def split_block(
        self, partition: Partition, block_idx: int
    ) -> Partition:
        """Split block at *block_idx* into two sub-blocks.

        Parameters
        ----------
        partition : Partition
            Current partition.
        block_idx : int
            Index of the block to split.

        Returns
        -------
        Partition
            New partition with the block split.
        """
        result, _ = self._split_block_impl(partition, block_idx)
        return result

    def merge_blocks(
        self, partition: Partition, idx1: int, idx2: int
    ) -> Partition:
        """Merge two adjacent blocks at *idx1* and *idx2*.

        Parameters
        ----------
        partition : Partition
            Current partition.
        idx1, idx2 : int
            Indices of adjacent blocks to merge.

        Returns
        -------
        Partition
            New partition with blocks merged.
        """
        result = partition.copy()
        lo, hi = min(idx1, idx2), max(idx1, idx2)
        merged = result.blocks[lo] + result.blocks[hi]
        result.blocks[lo] = merged
        del result.blocks[hi]
        return result

    # -----------------------------------------------------------------
    # Move implementations
    # -----------------------------------------------------------------

    def _split_move(self, partition: Partition) -> Tuple[Partition, float]:
        """Split a randomly chosen multi-element block into two.

        Returns
        -------
        Tuple[Partition, float]
            ``(proposed, move_ratio)``
        """
        splittable = [i for i, b in enumerate(partition.blocks) if len(b) > 1]
        if not splittable:
            return partition.copy(), 1.0

        block_idx = splittable[int(self._rng.integers(len(splittable)))]
        result, n_ways = self._split_block_impl(partition, block_idx)

        # Proposal ratio: probability of reverse (merge) / probability of forward (split)
        n_splittable = len(splittable)
        n_mergeable_after = result.n_blocks - 1
        move_ratio = n_splittable * n_ways / max(n_mergeable_after, 1)
        return result, move_ratio

    def _merge_move(self, partition: Partition) -> Tuple[Partition, float]:
        """Merge two randomly chosen adjacent blocks.

        Returns
        -------
        Tuple[Partition, float]
            ``(proposed, move_ratio)``
        """
        if partition.n_blocks < 2:
            return partition.copy(), 1.0

        idx = int(self._rng.integers(partition.n_blocks - 1))
        result = self.merge_blocks(partition, idx, idx + 1)

        # Proposal ratio
        n_mergeable = partition.n_blocks - 1
        n_splittable_after = sum(1 for b in result.blocks if len(b) > 1)
        merged_block = result.blocks[idx]
        n_split_ways = max(2 ** len(merged_block) - 2, 1)
        move_ratio = n_mergeable / max(n_splittable_after * n_split_ways, 1)
        return result, move_ratio

    def _swap_move(self, partition: Partition) -> Tuple[Partition, float]:
        """Swap one element between two adjacent blocks.

        Returns
        -------
        Tuple[Partition, float]
            ``(proposed, move_ratio)``
        """
        if partition.n_blocks < 2:
            return partition.copy(), 1.0

        result = partition.copy()
        pair_idx = int(self._rng.integers(partition.n_blocks - 1))
        block_a = result.blocks[pair_idx]
        block_b = result.blocks[pair_idx + 1]

        if not block_a or not block_b:
            return result, 1.0

        # Decide direction: move from A to B or B to A
        if self._rng.random() < 0.5 and len(block_a) > 1:
            # Move element from block A to block B
            elem_idx = int(self._rng.integers(len(block_a)))
            elem = block_a.pop(elem_idx)
            block_b.append(elem)
        elif len(block_b) > 1:
            # Move element from block B to block A
            elem_idx = int(self._rng.integers(len(block_b)))
            elem = block_b.pop(elem_idx)
            block_a.append(elem)
        elif len(block_a) > 1:
            elem_idx = int(self._rng.integers(len(block_a)))
            elem = block_a.pop(elem_idx)
            block_b.append(elem)
        else:
            # Both singletons — just swap
            block_a[0], block_b[0] = block_b[0], block_a[0]

        # Remove any empty blocks
        result.blocks = [b for b in result.blocks if b]
        return result, 1.0  # symmetric proposal

    def _split_block_impl(
        self, partition: Partition, block_idx: int
    ) -> Tuple[Partition, int]:
        """Implement block split at *block_idx*.

        Returns
        -------
        Tuple[Partition, int]
            ``(new_partition, n_ways)`` where n_ways is the number
            of possible split points for MH correction.
        """
        result = partition.copy()
        block = result.blocks[block_idx]
        n = len(block)
        if n < 2:
            return result, 1

        # Random split point
        split = int(self._rng.integers(1, n))
        self._rng.shuffle(block)
        left = block[:split]
        right = block[split:]

        result.blocks = (
            result.blocks[:block_idx]
            + [left, right]
            + result.blocks[block_idx + 1:]
        )
        n_ways = n - 1
        return result, n_ways

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def _score_partition(self, partition: Partition) -> float:
        """Score a partition as the best DAG score over compatible orderings.

        For each node, the parent candidates are all nodes in strictly
        preceding blocks plus all other nodes in the same block (since
        within a block, any ordering is valid, parents can come from
        the same block as well).

        We use the score where parents can come from earlier blocks only,
        which corresponds to the sum of best local scores.

        Parameters
        ----------
        partition : Partition
            A partition of nodes.

        Returns
        -------
        float
            Total log-score.
        """
        total = 0.0
        for block_idx, block in enumerate(partition.blocks):
            preds = partition.block_predecessors(block_idx)
            # Nodes in the same block can also be parents of each other
            # in some compatible ordering, so candidates include same-block
            # nodes and all preceding nodes
            same_block = set(block)
            for node in block:
                # Candidates: predecessors from earlier blocks + other nodes in same block
                candidates = preds + [n for n in block if n != node]
                best_score, _ = self._best_parents(node, candidates)
                total += best_score
        return total

    def _best_parents(
        self, node: int, candidates: List[int]
    ) -> Tuple[float, List[int]]:
        """Find best parent set for *node* from *candidates*.

        Uses exhaustive search for small candidate sets and greedy
        forward selection for larger ones.
        """
        cache_key = (node, frozenset(candidates))
        if cache_key in self._best_parent_cache:
            return self._best_parent_cache[cache_key]

        if len(candidates) > 15:
            result = self._greedy_parents(node, candidates)
        else:
            best_score = -math.inf
            best_parents: List[int] = []
            for k in range(min(self.max_parents, len(candidates)) + 1):
                for subset in itertools.combinations(candidates, k):
                    s = self._cache.local_score(node, list(subset))
                    if s > best_score:
                        best_score = s
                        best_parents = list(subset)
            result = (best_score, best_parents)

        self._best_parent_cache[cache_key] = result
        return result

    def _greedy_parents(
        self, node: int, candidates: List[int]
    ) -> Tuple[float, List[int]]:
        """Greedy forward selection of parents."""
        current_parents: List[int] = []
        current_score = self._cache.local_score(node, [])
        remaining = set(candidates)

        for _ in range(self.max_parents):
            best_add_score = current_score
            best_add: Optional[int] = None
            for c in remaining:
                trial = current_parents + [c]
                s = self._cache.local_score(node, trial)
                if s > best_add_score:
                    best_add_score = s
                    best_add = c
            if best_add is None:
                break
            current_parents.append(best_add)
            remaining.discard(best_add)
            current_score = best_add_score

        return current_score, current_parents

    # -----------------------------------------------------------------
    # Acceptance
    # -----------------------------------------------------------------

    def _acceptance_probability(
        self,
        new_score: float,
        old_score: float,
        proposal_ratio: float,
    ) -> float:
        """Metropolis-Hastings acceptance probability with proposal correction.

        Parameters
        ----------
        new_score : float
            Log-score of proposed partition.
        old_score : float
            Log-score of current partition.
        proposal_ratio : float
            ``q(current|proposed) / q(proposed|current)``.

        Returns
        -------
        float
            Acceptance probability in [0, 1].
        """
        log_ratio = new_score - old_score
        if proposal_ratio > 0:
            log_ratio += math.log(proposal_ratio)
        if log_ratio >= 0:
            return 1.0
        return math.exp(log_ratio)

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def _initial_partition(self) -> Partition:
        """Create the initial partition: each node in its own block.

        Returns
        -------
        Partition
        """
        nodes = list(self._rng.permutation(self.n_nodes))
        blocks = [[n] for n in nodes]
        return Partition(blocks=blocks)

    # -----------------------------------------------------------------
    # Conversion utilities
    # -----------------------------------------------------------------

    def partition_to_orders(self, partition: Partition) -> List[List[int]]:
        """Generate all linear orderings compatible with *partition*.

        Parameters
        ----------
        partition : Partition
            A partition of nodes.

        Returns
        -------
        List[List[int]]
            List of orderings consistent with the partition.
        """
        if not partition.blocks:
            return [[]]

        block_perms = [
            list(itertools.permutations(block)) for block in partition.blocks
        ]

        # Guard against combinatorial explosion
        total = 1
        for bp in block_perms:
            total *= len(bp)
            if total > 10000:
                # Sample rather than enumerate
                return self._sample_orders_from_partition(partition, 100)

        orders: List[List[int]] = []
        for combo in itertools.product(*block_perms):
            order: List[int] = []
            for perm in combo:
                order.extend(perm)
            orders.append(order)
        return orders

    def _sample_orders_from_partition(
        self, partition: Partition, n_samples: int
    ) -> List[List[int]]:
        """Sample random orderings from a partition."""
        orders: List[List[int]] = []
        for _ in range(n_samples):
            order: List[int] = []
            for block in partition.blocks:
                perm = list(self._rng.permutation(block))
                order.extend(perm)
            orders.append(order)
        return orders

    def sample_dag_from_partition(self, partition: Partition) -> NDArray:
        """Sample the optimal DAG for a random ordering from *partition*.

        Parameters
        ----------
        partition : Partition
            A partition of nodes.

        Returns
        -------
        NDArray
            Binary adjacency matrix.
        """
        order: List[int] = []
        for block in partition.blocks:
            perm = list(self._rng.permutation(block))
            order.extend(perm)

        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for pos, node in enumerate(order):
            candidates = list(order[:pos])
            _, best_parents = self._best_parents(node, candidates)
            for p in best_parents:
                adj[p, node] = 1.0
        return adj

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals accepted so far."""
        if self._total == 0:
            return 0.0
        return self._accepted / self._total

    @staticmethod
    def edge_posterior_from_partitions(
        partitions: List[Partition],
        sampler: "PartitionMCMC",
    ) -> NDArray:
        """Estimate edge posterior probabilities from partition samples.

        For each partition, samples a DAG and computes the empirical
        edge frequencies.
        """
        if not partitions:
            return np.empty((0, 0))
        n = sampler.n_nodes
        prob = np.zeros((n, n), dtype=np.float64)
        for part in partitions:
            dag = sampler.sample_dag_from_partition(part)
            prob += (dag != 0).astype(np.float64)
        prob /= len(partitions)
        return prob
