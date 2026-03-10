"""
Type definitions for tree decomposition and treewidth computation.

Provides data structures for representing tree decompositions, tree bags,
and elimination orderings used by the FPT dynamic-programming solver and
the moral-graph analysis routines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from causalcert.types import NodeId, NodeSet


# ---------------------------------------------------------------------------
# Tree bag
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeBag:
    """A single bag in a tree decomposition.

    Each bag contains a subset of vertices from the original graph and
    corresponds to a node in the decomposition tree.

    Attributes
    ----------
    bag_id : int
        Unique identifier for this bag within the decomposition.
    vertices : NodeSet
        Frozenset of vertex indices belonging to this bag.
    parent_id : int | None
        Bag id of the parent in the rooted tree, or ``None`` for the root.
    children_ids : tuple[int, ...]
        Bag ids of children in the rooted tree.
    """

    bag_id: int
    vertices: NodeSet
    parent_id: int | None = None
    children_ids: tuple[int, ...] = ()

    @property
    def width(self) -> int:
        """Width contribution of this bag (``|bag| - 1``)."""
        return len(self.vertices) - 1

    def contains(self, node: NodeId) -> bool:
        """Return ``True`` if *node* belongs to this bag."""
        return node in self.vertices

    def intersection(self, other: TreeBag) -> NodeSet:
        """Return the set of vertices shared with *other*.

        Parameters
        ----------
        other : TreeBag
            Another bag in the decomposition.

        Returns
        -------
        NodeSet
            Vertices present in both bags (the separator).
        """
        return self.vertices & other.vertices


# ---------------------------------------------------------------------------
# Elimination ordering
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EliminationOrdering:
    """A vertex elimination ordering for treewidth computation.

    An elimination ordering specifies the sequence in which vertices are
    eliminated from the graph.  The induced width of the ordering provides
    an upper bound on the treewidth.

    Attributes
    ----------
    order : tuple[NodeId, ...]
        Sequence of node ids in elimination order.
    induced_width : int
        Maximum number of neighbours of a vertex at the time of its
        elimination.  Equals the treewidth when the ordering is optimal.
    """

    order: tuple[NodeId, ...]
    induced_width: int

    def __post_init__(self) -> None:
        if len(self.order) != len(set(self.order)):
            raise ValueError("Elimination ordering contains duplicate nodes.")

    def position(self, node: NodeId) -> int:
        """Return the position of *node* in the ordering.

        Parameters
        ----------
        node : NodeId
            A vertex in the graph.

        Returns
        -------
        int
            Zero-based index into the ordering.

        Raises
        ------
        ValueError
            If *node* is not present in the ordering.
        """
        try:
            return self.order.index(node)
        except ValueError:
            raise ValueError(f"Node {node} not in elimination ordering.") from None


# ---------------------------------------------------------------------------
# Tree decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeDecomposition:
    """A tree decomposition of an undirected graph.

    A tree decomposition ``(T, {X_i})`` consists of a tree *T* whose nodes
    are bags *X_i* ⊆ V satisfying:

    1. Every vertex appears in at least one bag.
    2. For every edge ``(u, v)`` there is a bag containing both *u* and *v*.
    3. The bags containing any vertex *v* form a connected subtree of *T*.

    The width is ``max_i |X_i| - 1`` and the treewidth of *G* is the minimum
    width over all valid decompositions.

    Attributes
    ----------
    bags : tuple[TreeBag, ...]
        All bags in the decomposition, indexed by ``bag_id``.
    width : int
        Width of this decomposition (``max bag size − 1``).
    root_id : int
        Bag id of the root when the tree is rooted.
    n_vertices : int
        Number of vertices in the original graph.
    elimination_ordering : EliminationOrdering | None
        The elimination ordering that produced this decomposition, if available.
    """

    bags: tuple[TreeBag, ...]
    width: int
    root_id: int = 0
    n_vertices: int = 0
    elimination_ordering: EliminationOrdering | None = None

    def bag(self, bag_id: int) -> TreeBag:
        """Retrieve a bag by its identifier.

        Parameters
        ----------
        bag_id : int
            Unique bag identifier.

        Returns
        -------
        TreeBag

        Raises
        ------
        KeyError
            If no bag with that id exists.
        """
        for b in self.bags:
            if b.bag_id == bag_id:
                return b
        raise KeyError(f"No bag with id {bag_id}.")

    def bags_containing(self, node: NodeId) -> Sequence[TreeBag]:
        """Return all bags that contain *node*."""
        return [b for b in self.bags if b.contains(node)]

    @property
    def n_bags(self) -> int:
        """Number of bags in the decomposition."""
        return len(self.bags)
