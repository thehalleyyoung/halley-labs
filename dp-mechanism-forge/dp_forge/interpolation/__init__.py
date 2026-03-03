"""
Craig interpolation for DP mechanism synthesis and verification.

This package implements Craig interpolation techniques for extracting
abstractions and invariants useful in counterexample-guided synthesis.
Given two mutually inconsistent formulas A and B, a Craig interpolant I
satisfies: A ⊨ I, I ∧ B is unsatisfiable, and I only uses symbols
common to A and B.

In the DP context, interpolants are used to:
- Refine abstractions in CEGAR by extracting predicates that separate
  feasible mechanism states from privacy-violating states.
- Compute inductive invariants that certify DP guarantees.
- Guide synthesis by providing necessary conditions for DP feasibility.

Architecture:
    1. **InterpolantComputer** — Computes Craig interpolants from proof
       of unsatisfiability (via SMT solver or proof system).
    2. **SequenceInterpolator** — Computes sequence interpolants for
       multi-step privacy proofs (chain A₁,...,Aₙ where each
       consecutive pair is separated).
    3. **TreeInterpolator** — Computes tree interpolants for branching
       proof structures.
    4. **InterpolantSimplifier** — Simplifies interpolants for better
       predicate quality (fewer variables, smaller expressions).
    5. **PrivacyInterpolator** — Domain-specific interpolation for DP
       mechanism constraints.

Example::

    from dp_forge.interpolation import PrivacyInterpolator, InterpolantConfig

    interp = PrivacyInterpolator(config=InterpolantConfig())
    result = interp.compute_privacy_interpolant(
        mechanism_constraints=mech_formula,
        violation_constraints=violation_formula,
    )
    if result.success:
        print(f"Interpolant: {result.interpolant}")
        print(f"Strength: {result.strength}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    Formula,
    InterpolantType,
    Predicate,
    PrivacyBudget,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InterpolantStrength(Enum):
    """Strength classification of a Craig interpolant."""

    STRONGEST = auto()
    WEAKEST = auto()
    BALANCED = auto()

    def __repr__(self) -> str:
        return f"InterpolantStrength.{self.name}"


class SimplificationStrategy(Enum):
    """Strategy for simplifying interpolant formulas."""

    SUBSUMPTION = auto()
    VARIABLE_ELIMINATION = auto()
    QUANTIFIER_ELIMINATION = auto()
    NONE = auto()

    def __repr__(self) -> str:
        return f"SimplificationStrategy.{self.name}"


class ProofSystem(Enum):
    """Proof system used to derive interpolants."""

    RESOLUTION = auto()
    CUTTING_PLANES = auto()
    FARKAS_LEMMA = auto()
    HYPER_RESOLUTION = auto()

    def __repr__(self) -> str:
        return f"ProofSystem.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InterpolantConfig:
    """Configuration for interpolation computation.

    Attributes:
        interpolant_type: Type of interpolant to compute.
        strength: Desired interpolant strength (strongest vs. weakest).
        proof_system: Proof system for deriving interpolants.
        simplification: Simplification strategy for output interpolants.
        max_variables: Maximum number of variables in the interpolant.
        max_disjuncts: Maximum number of disjuncts in DNF interpolants.
        timeout_seconds: Maximum time for interpolant computation.
        verbose: Verbosity level.
    """

    interpolant_type: InterpolantType = InterpolantType.LINEAR_ARITHMETIC
    strength: InterpolantStrength = InterpolantStrength.BALANCED
    proof_system: ProofSystem = ProofSystem.FARKAS_LEMMA
    simplification: SimplificationStrategy = SimplificationStrategy.VARIABLE_ELIMINATION
    max_variables: int = 20
    max_disjuncts: int = 10
    timeout_seconds: float = 60.0
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.max_variables < 1:
            raise ValueError(f"max_variables must be >= 1, got {self.max_variables}")
        if self.max_disjuncts < 1:
            raise ValueError(f"max_disjuncts must be >= 1, got {self.max_disjuncts}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")

    def __repr__(self) -> str:
        return (
            f"InterpolantConfig(type={self.interpolant_type.name}, "
            f"strength={self.strength.name}, proof={self.proof_system.name})"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Interpolant:
    """A Craig interpolant between two formulas.

    An interpolant I for (A, B) satisfies:
    - A ⊨ I (I is implied by A)
    - I ∧ B is unsatisfiable (I is inconsistent with B)
    - vars(I) ⊆ vars(A) ∩ vars(B)

    Attributes:
        formula: The interpolant formula.
        interpolant_type: Type classification.
        common_variables: Variables shared between A and B.
        strength: Strength of this interpolant.
        proof_system: Proof system used to derive it.
    """

    formula: Formula
    interpolant_type: InterpolantType
    common_variables: FrozenSet[str]
    strength: InterpolantStrength = InterpolantStrength.BALANCED
    proof_system: Optional[ProofSystem] = None

    def __post_init__(self) -> None:
        extra_vars = self.formula.variables - self.common_variables
        if extra_vars:
            raise ValueError(
                f"Interpolant uses variables not in common set: {extra_vars}"
            )

    def as_predicate(self, name: Optional[str] = None) -> Predicate:
        """Convert this interpolant to a Predicate for CEGAR refinement."""
        pred_name = name or f"itp_{hash(self.formula.expr) % 10000}"
        return Predicate(name=pred_name, formula=self.formula, is_atomic=False)

    def __repr__(self) -> str:
        return (
            f"Interpolant(type={self.interpolant_type.name}, "
            f"vars={len(self.common_variables)}, "
            f"strength={self.strength.name})"
        )


@dataclass
class SequenceInterpolant:
    """A sequence of interpolants for a chain of formulas.

    Given formulas A₁, ..., Aₙ where A₁ ∧ ... ∧ Aₙ is unsatisfiable,
    sequence interpolants I₁, ..., Iₙ₋₁ satisfy:
    - A₁ ⊨ I₁
    - Iₖ ∧ Aₖ₊₁ ⊨ Iₖ₊₁ for k = 1, ..., n-2
    - Iₙ₋₁ ∧ Aₙ is unsatisfiable

    Attributes:
        interpolants: The sequence of interpolant formulas.
        formulas: The original formula chain.
        is_inductive: Whether the sequence forms an inductive proof.
    """

    interpolants: List[Interpolant]
    formulas: List[Formula]
    is_inductive: bool = False

    def __post_init__(self) -> None:
        if len(self.interpolants) != len(self.formulas) - 1:
            raise ValueError(
                f"Need {len(self.formulas) - 1} interpolants for "
                f"{len(self.formulas)} formulas, got {len(self.interpolants)}"
            )

    @property
    def length(self) -> int:
        """Length of the interpolant sequence."""
        return len(self.interpolants)

    def as_predicates(self) -> List[Predicate]:
        """Convert all interpolants to predicates for CEGAR."""
        return [itp.as_predicate(f"seq_itp_{k}") for k, itp in enumerate(self.interpolants)]

    def __repr__(self) -> str:
        inductive = ", inductive" if self.is_inductive else ""
        return f"SequenceInterpolant(length={self.length}{inductive})"


@dataclass
class TreeInterpolant:
    """A tree interpolant for branching proof structures.

    Attributes:
        root: The root interpolant formula.
        children: Child tree interpolants (subtrees).
        formula: The formula at this tree node.
    """

    root: Interpolant
    children: List[TreeInterpolant] = field(default_factory=list)
    formula: Optional[Formula] = None

    @property
    def depth(self) -> int:
        """Depth of the interpolant tree."""
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        """Total number of nodes in the tree."""
        return 1 + sum(c.size for c in self.children)

    def flatten(self) -> List[Interpolant]:
        """Flatten all interpolants in the tree into a list."""
        result = [self.root]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def __repr__(self) -> str:
        return f"TreeInterpolant(depth={self.depth}, size={self.size})"


@dataclass
class InterpolationResult:
    """Result of an interpolation computation.

    Attributes:
        success: Whether interpolation succeeded.
        interpolant: The computed interpolant (None if failed).
        computation_time: Time spent computing in seconds.
        proof_size: Size of the proof used to derive the interpolant.
        error_message: Error message if interpolation failed.
    """

    success: bool
    interpolant: Optional[Interpolant] = None
    computation_time: float = 0.0
    proof_size: int = 0
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.success and self.interpolant is None:
            raise ValueError("interpolant must be provided when success is True")
        if not self.success and self.error_message is None:
            raise ValueError("error_message must be provided when success is False")

    def __repr__(self) -> str:
        if self.success:
            return (
                f"InterpolationResult(success=True, "
                f"time={self.computation_time:.2f}s, proof_size={self.proof_size})"
            )
        return f"InterpolationResult(success=False, error={self.error_message!r})"


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class InterpolantEngine(Protocol):
    """Protocol for interpolation engine backends."""

    def compute(self, formula_a: Formula, formula_b: Formula) -> InterpolationResult:
        """Compute a Craig interpolant between A and B.

        Requires: A ∧ B is unsatisfiable.
        """
        ...

    def compute_sequence(self, formulas: List[Formula]) -> Optional[SequenceInterpolant]:
        """Compute sequence interpolants for a formula chain.

        Requires: A₁ ∧ ... ∧ Aₙ is unsatisfiable.
        """
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class InterpolantComputer:
    """Compute Craig interpolants from proofs of unsatisfiability.

    Uses the configured proof system to derive interpolants from
    unsatisfiability proofs, with optional simplification.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        from dp_forge.interpolation.craig import CraigInterpolant as _CI
        from dp_forge.interpolation.craig import SequenceInterpolation as _SI
        self._craig = _CI(self.config)
        self._seq = _SI(self.config)

    def compute(self, formula_a: Formula, formula_b: Formula) -> InterpolationResult:
        """Compute a Craig interpolant I such that A ⊨ I and I ∧ B is UNSAT.

        Args:
            formula_a: Formula A.
            formula_b: Formula B (must be inconsistent with A).

        Returns:
            InterpolationResult with the interpolant or error.
        """
        return self._craig.compute(formula_a, formula_b)

    def compute_sequence(
        self,
        formulas: List[Formula],
    ) -> Optional[SequenceInterpolant]:
        """Compute sequence interpolants for a formula chain.

        Args:
            formulas: Chain of formulas whose conjunction is UNSAT.

        Returns:
            SequenceInterpolant or None if computation fails.
        """
        return self._seq.compute(formulas)

    def compute_tree(
        self,
        root_formula: Formula,
        children: List[Tuple[Formula, List[Formula]]],
    ) -> Optional[TreeInterpolant]:
        """Compute tree interpolants for a branching proof.

        Args:
            root_formula: Formula at the root.
            children: List of (node_formula, subtree_formulas) pairs.

        Returns:
            TreeInterpolant or None if computation fails.
        """
        from dp_forge.interpolation.tree_interpolation import (
            TreeNode as _TN,
            TreeInterpolant as _TI,
        )
        # Build a tree from root + children
        root_node = _TN(node_id="root", formula=root_formula)
        for i, (node_f, subtree_fs) in enumerate(children):
            child = _TN(node_id=f"child_{i}", formula=node_f, parent=root_node)
            for j, sf in enumerate(subtree_fs):
                leaf = _TN(node_id=f"child_{i}_leaf_{j}", formula=sf, parent=child)
                child.children.append(leaf)
            root_node.children.append(child)
        ti = _TI(self.config)
        return ti.compute(root_node)


class InterpolantSimplifier:
    """Simplify interpolant formulas for predicate quality."""

    def __init__(
        self,
        strategy: SimplificationStrategy = SimplificationStrategy.VARIABLE_ELIMINATION,
    ) -> None:
        self.strategy = strategy

    def simplify(self, interpolant: Interpolant) -> Interpolant:
        """Simplify an interpolant formula.

        Args:
            interpolant: The interpolant to simplify.

        Returns:
            Simplified interpolant with fewer variables/disjuncts.
        """
        from dp_forge.interpolation.formula import (
            Formula as _F,
            Simplifier as _S,
            QuantifierElimination as _QE,
        )
        f = _F.from_dp_formula(interpolant.formula)
        simplified = _S().simplify(f)

        if self.strategy == SimplificationStrategy.VARIABLE_ELIMINATION:
            extra = simplified.variables - interpolant.common_variables
            if extra:
                simplified = _QE().eliminate(simplified, list(extra))
                simplified = _S().simplify(simplified)

        return Interpolant(
            formula=simplified.to_dp_formula(),
            interpolant_type=interpolant.interpolant_type,
            common_variables=interpolant.common_variables & simplified.variables,
            strength=interpolant.strength,
            proof_system=interpolant.proof_system,
        )

    def rank_predicates(
        self,
        predicates: List[Predicate],
        *,
        max_predicates: int = 10,
    ) -> List[Predicate]:
        """Rank and select the most useful predicates.

        Ranks by: fewer variables > shorter expression > alphabetical.
        """
        scored = []
        for p in predicates:
            n_vars = len(p.formula.variables)
            expr_len = len(p.formula.expr)
            scored.append((n_vars, expr_len, p.name, p))
        scored.sort()
        return [s[3] for s in scored[:max_predicates]]


class PrivacyInterpolator:
    """Domain-specific Craig interpolation for DP mechanism constraints.

    Specialises interpolation to the structure of DP constraint systems:
    probability simplex constraints, privacy ratio constraints, and
    utility objectives.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        from dp_forge.interpolation.privacy_interpolation import (
            PrivacyInterpolant as _PI,
            InductiveInterpolant as _II,
        )
        self._pi = _PI(self.config)
        self._ii = _II(self.config)

    def compute_privacy_interpolant(
        self,
        mechanism_constraints: Formula,
        violation_constraints: Formula,
    ) -> InterpolationResult:
        """Compute an interpolant separating feasible mechanisms from violations.

        Args:
            mechanism_constraints: Constraints defining the mechanism.
            violation_constraints: Constraints encoding a privacy violation.

        Returns:
            InterpolationResult with the separating interpolant.
        """
        return self._pi.compute(mechanism_constraints, violation_constraints)

    def compute_inductive_invariant(
        self,
        transition_formula: Formula,
        safety_property: Formula,
        budget: PrivacyBudget,
    ) -> Optional[Interpolant]:
        """Compute an inductive invariant certifying DP via interpolation.

        Args:
            transition_formula: Mechanism transition relation.
            safety_property: DP safety property to prove.
            budget: Privacy budget.

        Returns:
            Inductive invariant interpolant or None.
        """
        init_formula = Formula(
            expr=f"({safety_property.expr})",
            variables=safety_property.variables,
        )
        return self._ii.compute(init_formula, transition_formula, safety_property)

    def extract_predicates(
        self,
        interpolant: Interpolant,
        *,
        max_predicates: int = 20,
    ) -> List[Predicate]:
        """Extract atomic predicates from an interpolant for CEGAR.

        Args:
            interpolant: Source interpolant.
            max_predicates: Maximum number of predicates to extract.

        Returns:
            List of atomic predicates.
        """
        return self._pi.extract_predicates(
            interpolant, max_predicates=max_predicates,
        )


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def interpolate(
    formula_a: Formula,
    formula_b: Formula,
    *,
    config: Optional[InterpolantConfig] = None,
) -> InterpolationResult:
    """Convenience function for Craig interpolation.

    Args:
        formula_a: Formula A.
        formula_b: Formula B (A ∧ B must be UNSAT).
        config: Optional configuration.

    Returns:
        InterpolationResult with the interpolant.
    """
    from dp_forge.interpolation.craig import CraigInterpolant as _CI
    return _CI(config).compute(formula_a, formula_b)


def formula_to_predicates(
    formula: Formula,
    *,
    max_predicates: int = 20,
) -> List[Predicate]:
    """Decompose a formula into atomic predicates.

    Args:
        formula: Formula to decompose.
        max_predicates: Maximum number of predicates.

    Returns:
        List of atomic predicates.
    """
    from dp_forge.interpolation.formula import (
        Formula as _F,
        FormulaNode as _FN,
        NodeKind as _NK,
    )

    f = _F.from_dp_formula(formula)
    atoms: List[_FN] = []

    def _collect(n: _FN) -> None:
        if n.kind in (_NK.LEQ, _NK.EQ, _NK.LT, _NK.VAR):
            atoms.append(n)
            return
        if n.kind == _NK.CONST:
            return
        for c in n.children:
            _collect(c)

    _collect(f.node)
    predicates: List[Predicate] = []
    for i, atom in enumerate(atoms[:max_predicates]):
        atom_f = _F(atom)
        predicates.append(Predicate(
            name=f"pred_{i}",
            formula=atom_f.to_dp_formula(),
            is_atomic=True,
        ))
    return predicates


__all__ = [
    # Enums
    "InterpolantStrength",
    "SimplificationStrategy",
    "ProofSystem",
    # Config
    "InterpolantConfig",
    # Data types
    "Interpolant",
    "SequenceInterpolant",
    "TreeInterpolant",
    "InterpolationResult",
    # Protocols
    "InterpolantEngine",
    # Classes
    "InterpolantComputer",
    "InterpolantSimplifier",
    "PrivacyInterpolator",
    # Functions
    "interpolate",
    "formula_to_predicates",
]
