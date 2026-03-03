"""Specialized mutation operators for causal DAG manipulation.

Provides domain-specific mutations that target particular structural
motifs: v-structures, directed paths, local parent-set neighborhoods,
topological block swaps, skeleton modifications, and combined
structural-parametric mutations.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.operators.mutation import (
    MutationOperator,
    _can_reach,
    _has_cycle,
    _topological_sort,
    _topo_position,
)
from causal_qd.types import AdjacencyMatrix, DataMatrix, TopologicalOrder


# ---------------------------------------------------------------------------
# VStructureMutation
# ---------------------------------------------------------------------------


class VStructureMutation(MutationOperator):
    """Specifically add or remove v-structures (colliders).

    A v-structure is a triple ``i → k ← j`` where ``i`` and ``j``
    are not adjacent.  This operator randomly adds a new v-structure
    or removes an existing one.

    Parameters
    ----------
    add_prob : float
        Probability of adding a v-structure (vs. removing one).
        Default ``0.5``.
    """

    def __init__(self, add_prob: float = 0.5) -> None:
        self._add_prob = add_prob

    def mutate(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix:
        """Add or remove a random v-structure.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current DAG.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        AdjacencyMatrix
            Modified DAG.
        """
        result = dag.copy()
        n = dag.shape[0]

        if rng.random() < self._add_prob:
            return self._add_v_structure(result, n, rng)
        else:
            return self._remove_v_structure(result, n, rng)

    def _add_v_structure(
        self,
        adj: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Add a new v-structure i → k ← j.

        Picks a random target node k, then finds two non-adjacent nodes
        i, j that are not currently parents of k, adds both edges
        i → k and j → k (if acyclic), ensuring i and j stay non-adjacent.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG.
        n : int
            Number of nodes.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        AdjacencyMatrix
        """
        if n < 3:
            return adj

        # Pick random target node k
        k = rng.integers(0, n)
        current_parents = set(np.where(adj[:, k])[0])

        # Find non-parent, non-adjacent pairs
        candidates: List[Tuple[int, int]] = []
        for i in range(n):
            if i == k or i in current_parents:
                continue
            for j in range(i + 1, n):
                if j == k or j in current_parents:
                    continue
                # Ensure i and j are not adjacent
                if not adj[i, j] and not adj[j, i]:
                    candidates.append((i, j))

        if not candidates:
            return adj

        i, j = candidates[rng.integers(0, len(candidates))]

        # Try adding i → k
        adj[i, k] = 1
        if _has_cycle(adj):
            adj[i, k] = 0
            return adj

        # Try adding j → k
        adj[j, k] = 1
        if _has_cycle(adj):
            adj[j, k] = 0
            # Revert i → k too
            adj[i, k] = 0

        return adj

    def _remove_v_structure(
        self,
        adj: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Remove an existing v-structure by removing one of its edges.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG.
        n : int
            Number of nodes.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        AdjacencyMatrix
        """
        v_structures = self._find_v_structures(adj, n)
        if not v_structures:
            return adj

        idx = rng.integers(0, len(v_structures))
        i, k, j = v_structures[idx]

        # Remove one of the edges randomly
        if rng.random() < 0.5:
            adj[i, k] = 0
        else:
            adj[j, k] = 0

        return adj

    @staticmethod
    def _find_v_structures(
        adj: AdjacencyMatrix, n: int
    ) -> List[Tuple[int, int, int]]:
        """Find all v-structures in the DAG.

        Parameters
        ----------
        adj : AdjacencyMatrix
        n : int

        Returns
        -------
        List[Tuple[int, int, int]]
            (i, k, j) triples where i → k ← j and i ⊥ j.
        """
        result: List[Tuple[int, int, int]] = []
        for k in range(n):
            parents = list(np.where(adj[:, k])[0])
            for a in range(len(parents)):
                for b in range(a + 1, len(parents)):
                    i, j = parents[a], parents[b]
                    if not adj[i, j] and not adj[j, i]:
                        result.append((i, k, j))
        return result


# ---------------------------------------------------------------------------
# PathMutation
# ---------------------------------------------------------------------------


class PathMutation(MutationOperator):
    """Extend or shorten a randomly selected directed path.

    Randomly selects an existing directed path in the DAG and either
    extends it by adding an edge to an unconnected node, or shortens
    it by removing an intermediate edge.

    Parameters
    ----------
    extend_prob : float
        Probability of extending (vs. shortening) the path.
        Default ``0.5``.
    """

    def __init__(self, extend_prob: float = 0.5) -> None:
        self._extend_prob = extend_prob

    def mutate(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix:
        """Extend or shorten a directed path.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Current DAG.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        AdjacencyMatrix
        """
        result = dag.copy()
        n = dag.shape[0]

        if rng.random() < self._extend_prob:
            return self._extend_path(result, n, rng)
        else:
            return self._shorten_path(result, n, rng)

    def _extend_path(
        self,
        adj: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Extend a path by adding an edge at one end.

        Finds a sink node (out-degree 0 in the path) and tries to
        add an edge to a non-descendant node.

        Parameters
        ----------
        adj, n, rng

        Returns
        -------
        AdjacencyMatrix
        """
        # Find all sink nodes (no outgoing edges)
        out_deg = adj.sum(axis=1)
        sinks = np.where(out_deg == 0)[0]

        if len(sinks) == 0:
            # No sinks, try any edge addition
            sinks = np.arange(n)

        # Pick a random sink
        source = sinks[rng.integers(0, len(sinks))]

        # Find valid targets (not already reachable from source)
        targets = []
        for t in range(n):
            if t != source and not adj[source, t]:
                # Check if adding this edge is safe
                test = adj.copy()
                test[source, t] = 1
                if not _has_cycle(test):
                    targets.append(t)

        if targets:
            target = targets[rng.integers(0, len(targets))]
            adj[source, target] = 1

        return adj

    def _shorten_path(
        self,
        adj: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Shorten a path by removing an intermediate edge.

        Parameters
        ----------
        adj, n, rng

        Returns
        -------
        AdjacencyMatrix
        """
        edges = list(zip(*np.nonzero(adj)))
        if not edges:
            return adj

        # Pick a random edge to remove
        idx = rng.integers(0, len(edges))
        i, j = edges[idx]
        adj[i, j] = 0

        return adj


# ---------------------------------------------------------------------------
# NeighborhoodMutation
# ---------------------------------------------------------------------------


class NeighborhoodMutation(MutationOperator):
    """Modify the local parent set of a randomly chosen node.

    Selects a random node and modifies its parent set by adding,
    removing, or swapping one parent.

    Parameters
    ----------
    add_parent_prob : float
        Probability of adding a parent.  Default ``0.4``.
    remove_parent_prob : float
        Probability of removing a parent.  Default ``0.3``.
    swap_parent_prob : float
        Probability of swapping a parent (remove one, add another).
        Default ``0.3``.
    max_parents : int
        Maximum in-degree.  Default ``-1``.
    """

    def __init__(
        self,
        add_parent_prob: float = 0.4,
        remove_parent_prob: float = 0.3,
        swap_parent_prob: float = 0.3,
        max_parents: int = -1,
    ) -> None:
        total = add_parent_prob + remove_parent_prob + swap_parent_prob
        self._add_prob = add_parent_prob / total
        self._remove_prob = remove_parent_prob / total
        self._swap_prob = swap_parent_prob / total
        self._max_parents = max_parents

    def mutate(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix:
        """Modify the parent set of a random node.

        Parameters
        ----------
        dag, rng

        Returns
        -------
        AdjacencyMatrix
        """
        result = dag.copy()
        n = dag.shape[0]
        node = rng.integers(0, n)
        parents = list(np.where(result[:, node])[0])

        roll = rng.random()

        if roll < self._add_prob:
            return self._add_parent(result, node, parents, n, rng)
        elif roll < self._add_prob + self._remove_prob:
            return self._remove_parent(result, node, parents, rng)
        else:
            return self._swap_parent(result, node, parents, n, rng)

    def _add_parent(
        self,
        adj: AdjacencyMatrix,
        node: int,
        parents: List[int],
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Add a new parent to *node*."""
        if self._max_parents > 0 and len(parents) >= self._max_parents:
            return adj

        candidates = [
            i for i in range(n)
            if i != node and i not in parents and not _can_reach(adj, node, i)
        ]
        if candidates:
            new_parent = candidates[rng.integers(0, len(candidates))]
            adj[new_parent, node] = 1
        return adj

    def _remove_parent(
        self,
        adj: AdjacencyMatrix,
        node: int,
        parents: List[int],
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Remove a random parent from *node*."""
        if parents:
            old_parent = parents[rng.integers(0, len(parents))]
            adj[old_parent, node] = 0
        return adj

    def _swap_parent(
        self,
        adj: AdjacencyMatrix,
        node: int,
        parents: List[int],
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Remove one parent and add a different one."""
        if not parents:
            return self._add_parent(adj, node, parents, n, rng)

        # Remove a random parent
        old_parent = parents[rng.integers(0, len(parents))]
        adj[old_parent, node] = 0

        # Add a new parent
        new_parents = list(np.where(adj[:, node])[0])
        candidates = [
            i for i in range(n)
            if i != node and i not in new_parents and not _can_reach(adj, node, i)
        ]
        if candidates:
            new_parent = candidates[rng.integers(0, len(candidates))]
            adj[new_parent, node] = 1

        return adj


# ---------------------------------------------------------------------------
# BlockMutation
# ---------------------------------------------------------------------------


class BlockMutation(MutationOperator):
    """Swap blocks of nodes in the topological ordering.

    Divides the topological ordering into contiguous blocks and
    swaps two randomly chosen blocks, then re-orients edges to
    be consistent with the new ordering.

    Parameters
    ----------
    block_size_range : Tuple[int, int]
        Range of block sizes ``(min_size, max_size)``.
        Default ``(2, 4)``.
    """

    def __init__(
        self, block_size_range: Tuple[int, int] = (2, 4)
    ) -> None:
        self._min_block = block_size_range[0]
        self._max_block = block_size_range[1]

    def mutate(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix:
        """Swap two blocks in the topological order and reorient edges.

        Parameters
        ----------
        dag, rng

        Returns
        -------
        AdjacencyMatrix
        """
        n = dag.shape[0]
        if n < 4:
            return dag.copy()

        result = dag.copy()
        order = _topological_sort(result)

        # Choose block size
        block_size = rng.integers(
            self._min_block, min(self._max_block + 1, n // 2 + 1)
        )
        if block_size > n // 2:
            block_size = max(1, n // 2)

        # Choose two non-overlapping blocks
        max_start = n - block_size
        if max_start < 1:
            return result

        start1 = rng.integers(0, max_start)
        # Ensure non-overlapping
        valid_starts = [
            s for s in range(max_start + 1)
            if s + block_size <= start1 or s >= start1 + block_size
        ]
        if not valid_starts:
            return result

        start2 = valid_starts[rng.integers(0, len(valid_starts))]

        # Swap blocks in the ordering
        new_order = list(order)
        block1 = new_order[start1:start1 + block_size]
        block2 = new_order[start2:start2 + block_size]
        new_order[start1:start1 + block_size] = block2
        new_order[start2:start2 + block_size] = block1

        # Build position map
        pos = np.empty(n, dtype=int)
        for idx, node in enumerate(new_order):
            pos[node] = idx

        # Keep only edges consistent with new ordering
        new_adj = np.zeros_like(result)
        for i in range(n):
            for j in range(n):
                if result[i, j] and pos[i] < pos[j]:
                    new_adj[i, j] = 1

        return new_adj


# ---------------------------------------------------------------------------
# SkeletonMutation
# ---------------------------------------------------------------------------


class SkeletonMutation(MutationOperator):
    """Mutate the undirected skeleton while preserving edge orientations.

    Modifies which edges exist (the skeleton) without changing the
    direction of edges that remain.  New edges are oriented to be
    consistent with the current topological ordering.

    Parameters
    ----------
    add_skeleton_prob : float
        Probability of adding a skeleton edge.  Default ``0.5``.
    """

    def __init__(self, add_skeleton_prob: float = 0.5) -> None:
        self._add_prob = add_skeleton_prob

    def mutate(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix:
        """Add or remove an edge from the skeleton.

        New edges are oriented according to the topological order.
        Existing edges are removed without changing other orientations.

        Parameters
        ----------
        dag, rng

        Returns
        -------
        AdjacencyMatrix
        """
        result = dag.copy()
        n = dag.shape[0]

        if rng.random() < self._add_prob:
            return self._add_skeleton_edge(result, n, rng)
        else:
            return self._remove_skeleton_edge(result, n, rng)

    def _add_skeleton_edge(
        self,
        adj: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Add an edge oriented according to topological order."""
        skeleton = adj | adj.T
        pos = _topo_position(adj)

        # Find non-adjacent pairs
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if not skeleton[i, j]:
                    candidates.append((i, j))

        if not candidates:
            return adj

        idx = rng.integers(0, len(candidates))
        i, j = candidates[idx]

        # Orient according to topological order
        if pos[i] < pos[j]:
            adj[i, j] = 1
        else:
            adj[j, i] = 1

        # Verify acyclicity
        if _has_cycle(adj):
            if pos[i] < pos[j]:
                adj[i, j] = 0
            else:
                adj[j, i] = 0

        return adj

    def _remove_skeleton_edge(
        self,
        adj: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Remove a random edge from the skeleton."""
        edges = list(zip(*np.nonzero(adj)))
        if edges:
            idx = rng.integers(0, len(edges))
            i, j = edges[idx]
            adj[i, j] = 0
        return adj


# ---------------------------------------------------------------------------
# MixingMutation
# ---------------------------------------------------------------------------


class MixingMutation(MutationOperator):
    """Combine structural changes with parametric re-optimization.

    First applies a structural mutation (edge add/remove/reverse),
    then optionally re-evaluates and optimizes local parent sets
    by checking whether each edge is still beneficial given the
    new structure.

    Parameters
    ----------
    structural_op : MutationOperator | None
        The structural mutation operator.  Default: ``TopologicalMutation()``.
    optimization_prob : float
        Probability of performing the parametric clean-up after
        the structural mutation.  Default ``0.3``.
    score_fn : ScoreFunction | None
        Scoring function for optimization.  If ``None``, optimization
        is skipped.
    data : DataMatrix | None
        Data for scoring.  If ``None``, optimization is skipped.
    """

    def __init__(
        self,
        structural_op: Optional[MutationOperator] = None,
        optimization_prob: float = 0.3,
        score_fn: Optional[Any] = None,
        data: Optional[DataMatrix] = None,
    ) -> None:
        from causal_qd.operators.mutation import TopologicalMutation

        self._struct_op = structural_op or TopologicalMutation()
        self._opt_prob = optimization_prob
        self._score_fn = score_fn
        self._data = data

    def mutate(
        self, dag: AdjacencyMatrix, rng: np.random.Generator
    ) -> AdjacencyMatrix:
        """Apply structural mutation optionally followed by optimization.

        Parameters
        ----------
        dag, rng

        Returns
        -------
        AdjacencyMatrix
        """
        # Structural mutation
        result = self._struct_op.mutate(dag, rng)

        # Parametric optimization
        if (
            rng.random() < self._opt_prob
            and self._score_fn is not None
            and self._data is not None
        ):
            result = self._optimize_edges(result, rng)

        return result

    def _optimize_edges(
        self,
        adj: AdjacencyMatrix,
        rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Remove edges that don't improve the score.

        For each edge, temporarily remove it and check if the score
        improves.  If so, keep it removed.  Process edges in random
        order to avoid bias.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Current DAG.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        AdjacencyMatrix
            Optimized DAG.
        """
        result = adj.copy()
        n = adj.shape[0]
        score_fn = self._score_fn
        data = self._data

        if score_fn is None or data is None:
            return result

        base_score = score_fn.score(result, data)
        edges = list(zip(*np.nonzero(result)))
        rng.shuffle(edges)  # type: ignore[arg-type]

        for i, j in edges:
            result[i, j] = 0
            new_score = score_fn.score(result, data)
            if new_score <= base_score:
                result[i, j] = 1  # Revert: edge was beneficial
            else:
                base_score = new_score  # Accept removal

        return result
