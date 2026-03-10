"""
usability_oracle.bisimulation.models — Data structures for bisimulation analysis.

Provides:
  - :class:`Partition` — block partition of the state space.
  - :class:`BisimulationResult` — full result of a bisimulation computation.
  - :class:`CognitiveDistanceMatrix` — pairwise cognitive distances.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from usability_oracle.mdp.models import MDP


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    """A partition of the MDP state space into equivalence blocks.

    Each block is a ``frozenset[str]`` of state ids.  The partition is
    maintained as a list of blocks *and* a reverse index ``state_to_block``
    mapping each state id to its block index.

    Attributes
    ----------
    blocks : list[frozenset[str]]
        Ordered list of equivalence classes.
    state_to_block : dict[str, int]
        Maps state_id → index into *blocks*.
    """

    blocks: list[frozenset[str]] = field(default_factory=list)
    state_to_block: dict[str, int] = field(default_factory=dict)

    # ── Properties --------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks in the partition."""
        return len(self.blocks)

    # ── Query methods -----------------------------------------------------

    def get_block(self, state: str) -> frozenset[str]:
        """Return the equivalence class containing *state*.

        Raises
        ------
        KeyError
            If *state* is not in the partition.
        """
        idx = self.state_to_block.get(state)
        if idx is None:
            raise KeyError(f"State {state!r} not found in partition")
        return self.blocks[idx]

    def block_index(self, state: str) -> int:
        """Return the block index for *state*."""
        idx = self.state_to_block.get(state)
        if idx is None:
            raise KeyError(f"State {state!r} not found in partition")
        return idx

    def block_label(self, block_idx: int) -> str:
        """Return a canonical label for block *block_idx*."""
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise IndexError(f"Block index {block_idx} out of range [0, {len(self.blocks)})")
        members = sorted(self.blocks[block_idx])
        return f"B{block_idx}({','.join(members[:3])}{'…' if len(members) > 3 else ''})"

    def states(self) -> frozenset[str]:
        """Return all state ids in the partition."""
        all_states: set[str] = set()
        for block in self.blocks:
            all_states |= block
        return frozenset(all_states)

    # ── Mutation methods --------------------------------------------------

    def merge(self, block_a: int, block_b: int) -> "Partition":
        """Return a new Partition with blocks *block_a* and *block_b* merged.

        Parameters
        ----------
        block_a, block_b : int
            Indices of the blocks to merge.

        Returns
        -------
        Partition
            A new partition with one fewer block.
        """
        if block_a == block_b:
            return copy.deepcopy(self)
        lo, hi = min(block_a, block_b), max(block_a, block_b)
        merged = self.blocks[lo] | self.blocks[hi]
        new_blocks = []
        for i, b in enumerate(self.blocks):
            if i == lo:
                new_blocks.append(merged)
            elif i == hi:
                continue
            else:
                new_blocks.append(b)
        return Partition.from_blocks(new_blocks)

    def split(
        self,
        block_idx: int,
        criterion: Callable[[str], bool],
    ) -> "Partition":
        """Return a new Partition with *block_idx* split by *criterion*.

        States for which ``criterion(state)`` is True go into one new block;
        the rest form a second block.  If the split is trivial (one side
        empty), the partition is returned unchanged.

        Parameters
        ----------
        block_idx : int
            Index of the block to split.
        criterion : callable
            Predicate over state ids.

        Returns
        -------
        Partition
        """
        block = self.blocks[block_idx]
        yes = frozenset(s for s in block if criterion(s))
        no = block - yes
        if not yes or not no:
            return copy.deepcopy(self)
        new_blocks = list(self.blocks)
        new_blocks[block_idx] = yes
        new_blocks.append(no)
        return Partition.from_blocks(new_blocks)

    def refine(self, splitter: Callable[[frozenset[str]], list[frozenset[str]]]) -> "Partition":
        """Apply *splitter* to every block and collect the results.

        Parameters
        ----------
        splitter : callable
            Maps a block (frozenset of state ids) to a list of sub-blocks.

        Returns
        -------
        Partition
            The refined partition.
        """
        new_blocks: list[frozenset[str]] = []
        for block in self.blocks:
            sub_blocks = splitter(block)
            for sb in sub_blocks:
                if sb:
                    new_blocks.append(sb)
        return Partition.from_blocks(new_blocks)

    # ── Validation --------------------------------------------------------

    def is_valid(self) -> bool:
        """Check structural consistency of the partition.

        Returns True iff:
          1. Every state appears in exactly one block.
          2. ``state_to_block`` is consistent with ``blocks``.
          3. No block is empty.
        """
        seen: set[str] = set()
        for i, block in enumerate(self.blocks):
            if not block:
                return False
            for s in block:
                if s in seen:
                    return False
                seen.add(s)
                if self.state_to_block.get(s) != i:
                    return False
        return True

    # ── Factory -----------------------------------------------------------

    @classmethod
    def from_blocks(cls, blocks: list[frozenset[str]]) -> "Partition":
        """Create a Partition from a list of blocks, building the reverse index."""
        state_to_block: dict[str, int] = {}
        for idx, block in enumerate(blocks):
            for s in block:
                state_to_block[s] = idx
        return cls(blocks=list(blocks), state_to_block=state_to_block)

    @classmethod
    def trivial(cls, state_ids: list[str]) -> "Partition":
        """Return the single-block partition containing all states."""
        if not state_ids:
            return cls()
        return cls.from_blocks([frozenset(state_ids)])

    @classmethod
    def discrete(cls, state_ids: list[str]) -> "Partition":
        """Return the finest partition (one block per state)."""
        blocks = [frozenset([s]) for s in state_ids]
        return cls.from_blocks(blocks)

    # ── Comparison --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return set(self.blocks) == set(other.blocks)

    def __len__(self) -> int:
        return self.n_blocks

    def __repr__(self) -> str:
        return f"Partition(n_blocks={self.n_blocks}, n_states={len(self.state_to_block)})"


# ---------------------------------------------------------------------------
# BisimulationResult
# ---------------------------------------------------------------------------

@dataclass
class BisimulationResult:
    """Complete result of a bisimulation quotient computation.

    Attributes
    ----------
    partition : Partition
        The final equivalence-class partition.
    quotient_mdp : MDP
        The quotient (reduced) MDP whose states are blocks.
    abstraction_error : float
        Bound on the policy-value loss due to abstraction,
        measured as max_s |V(s) - V_abs(block(s))|.
    beta_used : float
        Rationality parameter β at which the bisimulation was computed.
    iterations : int
        Number of refinement iterations until convergence.
    refinement_history : list[int]
        Number of blocks at each refinement iteration.
    metadata : dict[str, Any]
        Additional diagnostic information.
    """

    partition: Partition
    quotient_mdp: MDP
    abstraction_error: float
    beta_used: float
    iterations: int
    refinement_history: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Ratio of abstract to concrete state-space sizes."""
        n_concrete = len(self.partition.state_to_block)
        if n_concrete == 0:
            return 1.0
        return self.partition.n_blocks / n_concrete

    def summary(self) -> str:
        """Return a one-line summary string."""
        return (
            f"Bisimulation: {len(self.partition.state_to_block)} states → "
            f"{self.partition.n_blocks} blocks "
            f"(β={self.beta_used:.2f}, error={self.abstraction_error:.4f}, "
            f"{self.iterations} iters)"
        )

    def __repr__(self) -> str:
        return (
            f"BisimulationResult(blocks={self.partition.n_blocks}, "
            f"error={self.abstraction_error:.4f}, β={self.beta_used:.2f})"
        )


# ---------------------------------------------------------------------------
# CognitiveDistanceMatrix
# ---------------------------------------------------------------------------

@dataclass
class CognitiveDistanceMatrix:
    """Pairwise cognitive distances between MDP states.

    The cognitive metric is defined as:

        d_cog(s₁, s₂) = sup_{β' ≤ β}  d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))

    where d_TV is total-variation distance and π_{β'} is the bounded-rational
    policy at rationality level β'.

    Attributes
    ----------
    distances : np.ndarray
        Symmetric n×n matrix of pairwise distances.
    state_ids : list[str]
        Ordered state identifiers corresponding to matrix rows/columns.
    """

    distances: np.ndarray
    state_ids: list[str]

    def _idx(self, state: str) -> int:
        """Return the matrix index for *state*."""
        try:
            return self.state_ids.index(state)
        except ValueError:
            raise KeyError(f"State {state!r} not in distance matrix")

    def distance(self, s1: str, s2: str) -> float:
        """Return d_cog(s1, s2).

        Parameters
        ----------
        s1, s2 : str
            State identifiers.

        Returns
        -------
        float
            The cognitive distance ∈ [0, 1].
        """
        i, j = self._idx(s1), self._idx(s2)
        return float(self.distances[i, j])

    def nearest_neighbors(self, state: str, k: int = 5) -> list[tuple[str, float]]:
        """Return the *k* nearest neighbors to *state* in cognitive distance.

        Parameters
        ----------
        state : str
            Query state identifier.
        k : int
            Number of neighbours to return.

        Returns
        -------
        list[tuple[str, float]]
            Sorted list of (state_id, distance) pairs, closest first.
        """
        i = self._idx(state)
        dists = self.distances[i, :]
        # Exclude self (distance 0)
        indices = np.argsort(dists)
        result: list[tuple[str, float]] = []
        for idx in indices:
            if self.state_ids[idx] == state:
                continue
            result.append((self.state_ids[idx], float(dists[idx])))
            if len(result) >= k:
                break
        return result

    def diameter(self) -> float:
        """Return the diameter of the metric space: max_{s1,s2} d_cog(s1, s2)."""
        if self.distances.size == 0:
            return 0.0
        return float(np.max(self.distances))

    def mean_distance(self) -> float:
        """Return the mean pairwise distance."""
        n = len(self.state_ids)
        if n < 2:
            return 0.0
        upper_triangle = self.distances[np.triu_indices(n, k=1)]
        return float(np.mean(upper_triangle))

    def threshold_partition(self, epsilon: float) -> Partition:
        """Create a partition by grouping states within distance *epsilon*.

        Uses single-linkage clustering: two states are in the same block
        iff they are connected by a chain of pairwise distances ≤ epsilon.

        Parameters
        ----------
        epsilon : float
            Distance threshold.

        Returns
        -------
        Partition
        """
        n = len(self.state_ids)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if self.distances[i, j] <= epsilon:
                    union(i, j)

        groups: dict[int, set[str]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, set()).add(self.state_ids[i])

        blocks = [frozenset(g) for g in groups.values()]
        return Partition.from_blocks(blocks)

    def __repr__(self) -> str:
        return (
            f"CognitiveDistanceMatrix(n_states={len(self.state_ids)}, "
            f"diameter={self.diameter():.4f})"
        )
