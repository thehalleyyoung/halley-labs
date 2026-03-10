"""
Algebra sub-package — algebraic operations on DAG edit sequences.

Provides data structures and interfaces for composing, canonicalising,
and ordering structural edits.  The edit lattice supports dominance
pruning during the robustness-radius search, and commutativity analysis
enables symmetry-aware branching in the CDCL and ILP solvers.
"""

from causalcert.algebra.types import (
    EditComposition,
    EditLattice,
    EditSequence,
)
from causalcert.algebra.protocols import (
    EditAlgebra,
)
from causalcert.algebra.edit_algebra import EditAlgebraImpl
from causalcert.algebra.lattice import EditLatticeImpl, HasseDiagram, build_hasse
from causalcert.algebra.equivalence import (
    EditEquivalence,
    EquivalenceClass,
    QuotientLattice,
    QuotientNode,
)

__all__ = [
    "EditSequence",
    "EditComposition",
    "EditLattice",
    # protocols
    "EditAlgebra",
    # implementations
    "EditAlgebraImpl",
    "EditLatticeImpl",
    "HasseDiagram",
    "build_hasse",
    "EditEquivalence",
    "EquivalenceClass",
    "QuotientLattice",
    "QuotientNode",
]
