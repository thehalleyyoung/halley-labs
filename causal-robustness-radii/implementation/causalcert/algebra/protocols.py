"""
Protocols for algebraic operations on DAG edit sequences.

Defines the structural sub-typing interface for the edit algebra, which
provides composition, canonicalisation, commutativity testing, and lattice
operations over sets of structural edits.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from causalcert.types import AdjacencyMatrix, StructuralEdit
from causalcert.algebra.types import EditComposition, EditLattice, EditSequence


# ---------------------------------------------------------------------------
# EditAlgebra
# ---------------------------------------------------------------------------


@runtime_checkable
class EditAlgebra(Protocol):
    """Algebraic operations on DAG edit sequences.

    The edit algebra provides:

    * **Composition** — combine two edit sequences into one, cancelling
      redundant edits (e.g., add-then-delete of the same edge).
    * **Canonicalisation** — reduce an edit sequence to a canonical
      (shortest, lexicographically ordered) representative.
    * **Commutativity test** — decide whether two edit sequences commute
      (produce the same result regardless of application order).
    * **Subsumption** — check whether one edit sequence subsumes another.
    * **Lattice construction** — build the edit lattice for dominance
      pruning during the robustness-radius search.
    """

    def compose(
        self,
        first: EditSequence,
        second: EditSequence,
        adj: AdjacencyMatrix,
    ) -> EditComposition:
        """Compose two edit sequences applied to *adj*.

        Applies *first* then *second* and returns a canonicalised
        composition that cancels redundant edits (e.g., an add followed
        by a delete of the same edge).

        Parameters
        ----------
        first : EditSequence
            Edit sequence applied first.
        second : EditSequence
            Edit sequence applied after *first*.
        adj : AdjacencyMatrix
            The DAG to which the edits are applied.

        Returns
        -------
        EditComposition
            Composition object with canonical form and commutativity flag.
        """
        ...

    def canonicalise(
        self,
        seq: EditSequence,
        adj: AdjacencyMatrix,
    ) -> EditSequence:
        """Reduce an edit sequence to canonical form.

        The canonical form is the shortest equivalent sequence with edits
        ordered lexicographically by ``(source, target, edit_type)``.
        Redundant pairs are cancelled.

        Parameters
        ----------
        seq : EditSequence
            An edit sequence to canonicalise.
        adj : AdjacencyMatrix
            The DAG providing context for redundancy detection.

        Returns
        -------
        EditSequence
            Canonical representative of the equivalence class.
        """
        ...

    def commutes(
        self,
        a: EditSequence,
        b: EditSequence,
        adj: AdjacencyMatrix,
    ) -> bool:
        """Test whether two edit sequences commute on *adj*.

        Two sequences commute if ``apply(apply(adj, a), b)`` produces the
        same graph as ``apply(apply(adj, b), a)``.

        Parameters
        ----------
        a : EditSequence
            First edit sequence.
        b : EditSequence
            Second edit sequence.
        adj : AdjacencyMatrix
            The DAG on which commutativity is tested.

        Returns
        -------
        bool
            ``True`` if *a* and *b* commute.
        """
        ...

    def subsumes(
        self,
        sub: EditSequence,
        sup: EditSequence,
    ) -> bool:
        """Test whether *sub* is subsumed by *sup*.

        An edit sequence *sub* is subsumed by *sup* if the affected edges
        of *sub* are a subset of those of *sup*.

        Parameters
        ----------
        sub : EditSequence
            Candidate subset sequence.
        sup : EditSequence
            Candidate superset sequence.

        Returns
        -------
        bool
            ``True`` if every edge affected by *sub* is also affected by
            *sup*.
        """
        ...

    def build_lattice(
        self,
        candidates: Sequence[EditSequence],
        adj: AdjacencyMatrix,
    ) -> EditLattice:
        """Build an edit lattice from a collection of candidate sequences.

        The lattice orders sequences by the subset relation on their
        affected edge sets and identifies minimal overturning elements.

        Parameters
        ----------
        candidates : Sequence[EditSequence]
            Edit sequences to insert into the lattice.
        adj : AdjacencyMatrix
            The original DAG providing structural context.

        Returns
        -------
        EditLattice
            Lattice structure for dominance pruning.
        """
        ...

    def inverse(
        self,
        seq: EditSequence,
    ) -> EditSequence:
        """Compute the inverse of an edit sequence.

        The inverse undoes each edit in reverse order: adds become deletes,
        deletes become adds, and reverses are self-inverse.

        Parameters
        ----------
        seq : EditSequence
            Edit sequence to invert.

        Returns
        -------
        EditSequence
            An edit sequence such that composing *seq* with its inverse
            yields the identity (no net change).
        """
        ...
