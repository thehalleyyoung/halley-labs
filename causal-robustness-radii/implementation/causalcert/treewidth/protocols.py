"""
Protocols for tree decomposition and treewidth algorithms.

Defines structural sub-typing interfaces for decomposition algorithms and
bag processors used by the FPT dynamic-programming solver (ALG 7).
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet, TreewidthBound
from causalcert.treewidth.types import EliminationOrdering, TreeBag, TreeDecomposition


# ---------------------------------------------------------------------------
# DecompositionAlgorithm
# ---------------------------------------------------------------------------


@runtime_checkable
class DecompositionAlgorithm(Protocol):
    """Algorithm that computes a tree decomposition of a graph.

    Implementations may use exact methods (e.g., BT-based search) or
    heuristics (e.g., min-degree, min-fill) to construct the decomposition.
    """

    def decompose(
        self,
        adj: AdjacencyMatrix,
        *,
        upper_bound: int | None = None,
    ) -> TreeDecomposition:
        """Compute a tree decomposition of the graph.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix of the (moral) graph.  For directed graphs,
            callers should first compute the moral graph.
        upper_bound : int | None, optional
            If provided, the algorithm may prune branches whose width would
            exceed this bound.

        Returns
        -------
        TreeDecomposition
            A valid tree decomposition.

        Raises
        ------
        RuntimeError
            If the decomposition could not be computed within resource limits.
        """
        ...

    def elimination_ordering(
        self,
        adj: AdjacencyMatrix,
    ) -> EliminationOrdering:
        """Compute a (heuristic) elimination ordering for the graph.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix of the graph.

        Returns
        -------
        EliminationOrdering
            A vertex elimination ordering with its induced width.
        """
        ...

    def treewidth_bounds(
        self,
        adj: AdjacencyMatrix,
    ) -> TreewidthBound:
        """Compute lower and upper bounds on the treewidth.

        Parameters
        ----------
        adj : AdjacencyMatrix
            Adjacency matrix of the graph.

        Returns
        -------
        TreewidthBound
            Lower bound, upper bound, and whether the result is exact.
        """
        ...


# ---------------------------------------------------------------------------
# BagProcessor
# ---------------------------------------------------------------------------


@runtime_checkable
class BagProcessor(Protocol):
    """Processes tree bags during bottom-up dynamic programming.

    The FPT solver traverses the tree decomposition bottom-up, calling
    :meth:`process_leaf`, :meth:`process_introduce`, :meth:`process_forget`,
    and :meth:`process_join` at the appropriate bags.  Each method computes
    a partial DP table for the sub-tree rooted at that bag.
    """

    def process_leaf(
        self,
        bag: TreeBag,
    ) -> dict[NodeSet, Any]:
        """Initialize the DP table for a leaf bag.

        Parameters
        ----------
        bag : TreeBag
            A leaf bag containing a single vertex.

        Returns
        -------
        dict[NodeSet, Any]
            DP table mapping feasible edit subsets to objective values.
        """
        ...

    def process_introduce(
        self,
        bag: TreeBag,
        introduced: NodeId,
        child_table: dict[NodeSet, Any],
    ) -> dict[NodeSet, Any]:
        """Extend the DP table when a vertex is introduced.

        Parameters
        ----------
        bag : TreeBag
            The introduce bag.
        introduced : NodeId
            The vertex being introduced (present in *bag* but not in child).
        child_table : dict[NodeSet, Any]
            DP table from the child bag.

        Returns
        -------
        dict[NodeSet, Any]
            Updated DP table incorporating the new vertex.
        """
        ...

    def process_forget(
        self,
        bag: TreeBag,
        forgotten: NodeId,
        child_table: dict[NodeSet, Any],
    ) -> dict[NodeSet, Any]:
        """Project out a vertex from the DP table.

        Parameters
        ----------
        bag : TreeBag
            The forget bag.
        forgotten : NodeId
            The vertex being forgotten (present in child but not in *bag*).
        child_table : dict[NodeSet, Any]
            DP table from the child bag.

        Returns
        -------
        dict[NodeSet, Any]
            DP table after marginalising over *forgotten*.
        """
        ...

    def process_join(
        self,
        bag: TreeBag,
        left_table: dict[NodeSet, Any],
        right_table: dict[NodeSet, Any],
    ) -> dict[NodeSet, Any]:
        """Merge DP tables from two children at a join bag."""
        ...
