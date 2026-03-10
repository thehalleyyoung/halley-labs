"""
Type definitions for the algebraic operations on DAG edits sub-package.

Provides data structures for representing sequences, compositions, and
lattice structures over structural edits.  These enable reasoning about
edit-set relationships which the solver exploits for symmetry breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    EdgeTuple,
    NodeId,
    NodeSet,
    StructuralEdit,
)


# ---------------------------------------------------------------------------
# EditSequence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditSequence:
    """An ordered sequence of structural edits applied to a DAG.

    Edit sequences are the fundamental objects of the edit algebra.
    Two sequences may produce the same graph when applied, in which case
    they are *equivalent*.

    Attributes
    ----------
    edits : tuple[StructuralEdit, ...]
        Edits in application order.
    source_hash : int | None
        Hash of the DAG to which this sequence is intended to be applied.
        Used for safety checks when composing sequences.
    """

    edits: tuple[StructuralEdit, ...]
    source_hash: int | None = None

    @property
    def cost(self) -> int:
        """Total cost of the edit sequence (sum of unit costs)."""
        return sum(e.cost for e in self.edits)

    @property
    def length(self) -> int:
        """Number of edits in the sequence."""
        return len(self.edits)

    @property
    def affected_edges(self) -> frozenset[EdgeTuple]:
        """Set of edges touched by any edit in this sequence."""
        return frozenset(e.edge for e in self.edits)

    @property
    def affected_nodes(self) -> NodeSet:
        """Set of nodes involved in at least one edit."""
        nodes: set[NodeId] = set()
        for e in self.edits:
            nodes.add(e.source)
            nodes.add(e.target)
        return frozenset(nodes)

    def __iter__(self) -> Iterator[StructuralEdit]:
        """Iterate over the edits in order."""
        return iter(self.edits)

    def __len__(self) -> int:
        return len(self.edits)

    def is_disjoint(self, other: EditSequence) -> bool:
        """Check whether two sequences affect disjoint edge sets."""
        return self.affected_edges.isdisjoint(other.affected_edges)


# ---------------------------------------------------------------------------
# EditComposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EditComposition:
    """Composition of two edit sequences applied consecutively.

    Represents the result of first applying *first* and then *second*
    to a DAG.  The composition tracks provenance so that decomposition
    into the original parts is always possible.

    Attributes
    ----------
    first : EditSequence
        The edit sequence applied first.
    second : EditSequence
        The edit sequence applied after *first*.
    canonical : EditSequence
        A canonicalised (merged, de-duplicated) version of the full
        composition.  Redundant edit pairs (e.g., add then delete of the
        same edge) are cancelled.
    commutes : bool
        ``True`` if applying *second* before *first* yields the same graph.
    """

    first: EditSequence
    second: EditSequence
    canonical: EditSequence
    commutes: bool = False

    @property
    def total_cost(self) -> int:
        """Cost of the canonical (simplified) sequence."""
        return self.canonical.cost

    @property
    def raw_cost(self) -> int:
        """Sum of costs before cancellation."""
        return self.first.cost + self.second.cost

    @property
    def cancellation_savings(self) -> int:
        """Number of edits cancelled during canonicalisation."""
        return self.raw_cost - self.total_cost


# ---------------------------------------------------------------------------
# EditLattice
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EditLattice:
    """Lattice of edit sets ordered by the subset relation.

    The edit lattice organises all explored edit sets into a partially
    ordered structure where ``A ≤ B`` iff the edges affected by *A* are a
    subset of those affected by *B*.  The lattice enables:

    * **Dominance pruning** — if edit set *A* already overturns the
      conclusion and *A ≤ B*, then *B* need not be explored.
    * **Minimal element extraction** — the minimal overturning sets are
      the robustness-radius witnesses.

    Attributes
    ----------
    elements : list[EditSequence]
        All edit sequences inserted into the lattice.
    overturning : list[EditSequence]
        Subset of elements that overturn the causal conclusion.
    minimal_elements : list[EditSequence]
        Overturning elements with no proper overturning subset.
    """

    elements: list[EditSequence] = field(default_factory=list)
    overturning: list[EditSequence] = field(default_factory=list)
    minimal_elements: list[EditSequence] = field(default_factory=list)

    def insert(self, seq: EditSequence, *, overturns: bool = False) -> None:
        """Insert an edit sequence into the lattice.

        Parameters
        ----------
        seq : EditSequence
            The edit sequence to insert.
        overturns : bool, optional
            Whether this sequence overturns the causal conclusion.
        """
        self.elements.append(seq)
        if overturns:
            self.overturning.append(seq)
            self._update_minimal(seq)

    def _update_minimal(self, seq: EditSequence) -> None:
        """Re-compute minimal elements after inserting an overturning set."""
        edges = seq.affected_edges
        self.minimal_elements = [
            m for m in self.minimal_elements
            if not edges < m.affected_edges
        ]
        if not any(m.affected_edges <= edges for m in self.minimal_elements):
            self.minimal_elements.append(seq)

    def dominates(self, seq: EditSequence) -> bool:
        """Check whether *seq* is dominated by an existing minimal element."""
        edges = seq.affected_edges
        return any(m.affected_edges <= edges for m in self.minimal_elements)

    @property
    def robustness_radius(self) -> int | None:
        """Minimum cost among overturning elements, or ``None`` if empty."""
        if not self.minimal_elements:
            return None
        return min(m.cost for m in self.minimal_elements)
