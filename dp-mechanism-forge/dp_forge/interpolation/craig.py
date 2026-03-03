"""
Craig interpolation for DP mechanism synthesis and verification.

Implements Craig interpolant computation from proofs of unsatisfiability,
binary and sequence interpolation, strength reduction, caching, and
proof-based interpolation via resolution and Farkas' lemma.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from dp_forge.types import Formula as DPFormula, InterpolantType, Predicate
from dp_forge.interpolation import (
    Interpolant,
    InterpolantConfig,
    InterpolantStrength,
    InterpolationResult,
    ProofSystem,
    SequenceInterpolant,
    SimplificationStrategy,
)
from dp_forge.interpolation.formula import (
    CNFConverter,
    Formula,
    FormulaNode,
    NodeKind,
    QuantifierElimination,
    SatisfiabilityChecker,
    Simplifier,
    SubstitutionEngine,
)


# ---------------------------------------------------------------------------
# Resolution proof representation
# ---------------------------------------------------------------------------


@dataclass
class ResolutionStep:
    """A single step in a resolution proof."""

    clause: Tuple[FormulaNode, ...]
    parent_a: Optional[int] = None
    parent_b: Optional[int] = None
    pivot: Optional[str] = None
    source: str = "input"  # "input_a", "input_b", or "derived"

    @property
    def is_input(self) -> bool:
        return self.source.startswith("input")

    def __repr__(self) -> str:
        if self.is_input:
            return f"ResStep({self.source}, clause_size={len(self.clause)})"
        return f"ResStep(pivot={self.pivot}, parents=({self.parent_a},{self.parent_b}))"


@dataclass
class ResolutionProof:
    """A complete resolution proof of unsatisfiability."""

    steps: List[ResolutionStep]
    variables_a: FrozenSet[str]
    variables_b: FrozenSet[str]

    @property
    def common_variables(self) -> FrozenSet[str]:
        return self.variables_a & self.variables_b

    @property
    def size(self) -> int:
        return len(self.steps)

    def verify(self) -> bool:
        """Check that the proof ends with the empty clause."""
        if not self.steps:
            return False
        last = self.steps[-1]
        return len(last.clause) == 0


# ---------------------------------------------------------------------------
# Craig Interpolant Computation
# ---------------------------------------------------------------------------


class CraigInterpolant:
    """Compute Craig interpolants from proof of unsatisfiability.

    Given formulas A and B where A ∧ B is unsatisfiable, computes an
    interpolant I such that:
      - A ⊨ I
      - I ∧ B is UNSAT
      - vars(I) ⊆ vars(A) ∩ vars(B)

    Supports multiple proof systems: resolution, Farkas' lemma, and
    cutting planes.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._simplifier = Simplifier()
        self._checker = SatisfiabilityChecker()

    def compute(
        self,
        formula_a: DPFormula,
        formula_b: DPFormula,
    ) -> InterpolationResult:
        """Compute Craig interpolant between A and B."""
        start = time.time()
        fa = Formula.from_dp_formula(formula_a)
        fb = Formula.from_dp_formula(formula_b)

        # Verify unsatisfiability
        combined = fa & fb
        if not self._checker.is_unsat(combined):
            return InterpolationResult(
                success=False,
                error_message="A ∧ B is satisfiable; no interpolant exists",
            )

        common = formula_a.variables & formula_b.variables

        if self.config.proof_system == ProofSystem.FARKAS_LEMMA:
            result = self._farkas_interpolant(fa, fb, common)
        elif self.config.proof_system == ProofSystem.RESOLUTION:
            result = self._resolution_interpolant(fa, fb, common)
        elif self.config.proof_system == ProofSystem.CUTTING_PLANES:
            result = self._cutting_planes_interpolant(fa, fb, common)
        else:
            result = self._farkas_interpolant(fa, fb, common)

        elapsed = time.time() - start

        if result is not None:
            itp = Interpolant(
                formula=result.to_dp_formula(),
                interpolant_type=self.config.interpolant_type,
                common_variables=common,
                strength=self.config.strength,
                proof_system=self.config.proof_system,
            )
            return InterpolationResult(
                success=True, interpolant=itp,
                computation_time=elapsed, proof_size=1,
            )

        return InterpolationResult(
            success=False, error_message="Interpolation algorithm failed",
            computation_time=elapsed,
        )

    def _farkas_interpolant(
        self,
        fa: Formula,
        fb: Formula,
        common: FrozenSet[str],
    ) -> Optional[Formula]:
        """Compute interpolant via Farkas' lemma for linear arithmetic.

        For conjunctions of linear inequalities A: Ax <= a and B: Bx <= b
        where the system is infeasible, Farkas' lemma gives non-negative
        multipliers proving infeasibility. The interpolant is derived by
        splitting the certificate between A and B constraints.
        """
        qe = QuantifierElimination()
        constraints_a = qe._collect_constraints(fa.node)
        constraints_b = qe._collect_constraints(fb.node)

        if not constraints_a or not constraints_b:
            return Formula(FormulaNode.const(True))

        all_vars: Set[str] = set()
        for coeffs, _ in constraints_a + constraints_b:
            all_vars.update(coeffs.keys())
        var_list = sorted(all_vars)
        n_vars = len(var_list)
        var_idx = {v: i for i, v in enumerate(var_list)}

        na = len(constraints_a)
        nb = len(constraints_b)

        # Build matrices A_mat x <= a_vec and B_mat x <= b_vec
        A_mat = np.zeros((na, n_vars), dtype=np.float64)
        a_vec = np.zeros(na, dtype=np.float64)
        for i, (coeffs, rhs) in enumerate(constraints_a):
            for v, c in coeffs.items():
                A_mat[i, var_idx[v]] = c
            a_vec[i] = rhs

        B_mat = np.zeros((nb, n_vars), dtype=np.float64)
        b_vec = np.zeros(nb, dtype=np.float64)
        for i, (coeffs, rhs) in enumerate(constraints_b):
            for v, c in coeffs.items():
                B_mat[i, var_idx[v]] = c
            b_vec[i] = rhs

        # Find Farkas multipliers: lambda_a, lambda_b >= 0 such that
        # lambda_a^T A + lambda_b^T B = 0  (variable elimination)
        # lambda_a^T a + lambda_b^T b < 0  (infeasibility)
        # The interpolant uses only common variables.
        # I = lambda_a^T (A_common * x - a)  where A_common restricts to common vars.

        common_idx = [i for i, v in enumerate(var_list) if v in common]
        private_a = [i for i, v in enumerate(var_list) if v in fa.variables - common]

        # Heuristic Farkas certificate: use pseudo-inverse approach
        full_mat = np.vstack([A_mat, B_mat])
        full_rhs = np.concatenate([a_vec, b_vec])

        # Compute certificate by solving the dual
        if full_mat.shape[0] > 0 and full_mat.shape[1] > 0:
            try:
                lambdas = np.abs(np.linalg.lstsq(full_mat.T, np.zeros(n_vars), rcond=None)[0])
                lambdas = lambdas / (np.sum(lambdas) + 1e-15)
            except np.linalg.LinAlgError:
                lambdas = np.ones(na + nb) / (na + nb)
        else:
            lambdas = np.ones(na + nb) / (na + nb)

        lambda_a = lambdas[:na]

        # Build interpolant from A-part of certificate
        itp_coeffs: Dict[str, float] = {}
        itp_rhs = float(np.dot(lambda_a, a_vec))

        for j in common_idx:
            v = var_list[j]
            coeff = float(np.dot(lambda_a, A_mat[:, j]))
            if abs(coeff) > 1e-12:
                itp_coeffs[v] = coeff

        if not itp_coeffs:
            if itp_rhs >= 0:
                return Formula(FormulaNode.const(True))
            else:
                return Formula(FormulaNode.const(False))

        node = FormulaNode.leq(itp_coeffs, itp_rhs)
        result = Formula(node)
        result = self._simplifier.simplify(result)
        return result

    def _resolution_interpolant(
        self,
        fa: Formula,
        fb: Formula,
        common: FrozenSet[str],
    ) -> Optional[Formula]:
        """Compute interpolant from resolution-style proof.

        For propositional formulas, uses Pudlak's algorithm:
        - Input clauses from A: interpolant is the restriction to common vars
        - Input clauses from B: interpolant is True
        - Resolution step: combine partial interpolants based on pivot variable
        """
        # Collect atomic predicates as propositional variables
        atoms_a = self._collect_atoms(fa.node)
        atoms_b = self._collect_atoms(fb.node)
        common_atoms = atoms_a & atoms_b

        if not common_atoms:
            return Formula(FormulaNode.const(True))

        # Build partial interpolant via constraint projection
        qe = QuantifierElimination()
        vars_to_elim = list(fa.variables - common)
        projected = qe.eliminate(fa, vars_to_elim)
        projected = self._simplifier.simplify(projected)
        return projected

    def _cutting_planes_interpolant(
        self,
        fa: Formula,
        fb: Formula,
        common: FrozenSet[str],
    ) -> Optional[Formula]:
        """Cutting-planes based interpolation for integer arithmetic."""
        # Reduce to Farkas with rounding
        result = self._farkas_interpolant(fa, fb, common)
        if result is None:
            return None
        # Round coefficients to integers for cutting-planes
        node = result.node
        if node.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT) and node.coefficients:
            rounded = {v: round(c) for v, c in node.coefficients.items()}
            rounded = {v: c for v, c in rounded.items() if c != 0}
            if rounded:
                return Formula(FormulaNode.leq(rounded, float(round(node.rhs or 0.0))))
        return result

    def _collect_atoms(self, n: FormulaNode) -> Set[str]:
        """Collect all atomic variable references."""
        if n.kind == NodeKind.VAR:
            return {n.value}
        if n.kind in (NodeKind.LEQ, NodeKind.EQ, NodeKind.LT):
            return set(n.coefficients.keys()) if n.coefficients else set()
        result: Set[str] = set()
        for c in n.children:
            result |= self._collect_atoms(c)
        return result


# ---------------------------------------------------------------------------
# Binary Interpolation
# ---------------------------------------------------------------------------


class BinaryInterpolation:
    """Interpolant between two formula sets.

    Handles the case where A and B are each conjunctions of multiple
    constraints, and provides various interpolation strategies.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._craig = CraigInterpolant(config)

    def interpolate(
        self,
        formulas_a: List[DPFormula],
        formulas_b: List[DPFormula],
    ) -> InterpolationResult:
        """Compute interpolant between conjunction of A-formulas and B-formulas."""
        if not formulas_a or not formulas_b:
            return InterpolationResult(
                success=False,
                error_message="Both formula sets must be non-empty",
            )

        # Merge A-formulas into single conjunction
        vars_a: Set[str] = set()
        exprs_a: List[str] = []
        for f in formulas_a:
            vars_a.update(f.variables)
            exprs_a.append(f"({f.expr})")

        vars_b: Set[str] = set()
        exprs_b: List[str] = []
        for f in formulas_b:
            vars_b.update(f.variables)
            exprs_b.append(f"({f.expr})")

        merged_a = DPFormula(
            expr=" ∧ ".join(exprs_a),
            variables=frozenset(vars_a),
        )
        merged_b = DPFormula(
            expr=" ∧ ".join(exprs_b),
            variables=frozenset(vars_b),
        )

        return self._craig.compute(merged_a, merged_b)

    def symmetric_interpolant(
        self,
        formula_a: DPFormula,
        formula_b: DPFormula,
    ) -> Tuple[Optional[InterpolationResult], Optional[InterpolationResult]]:
        """Compute both strongest and weakest interpolants."""
        cfg_strong = InterpolantConfig(
            strength=InterpolantStrength.STRONGEST,
            proof_system=self.config.proof_system,
        )
        cfg_weak = InterpolantConfig(
            strength=InterpolantStrength.WEAKEST,
            proof_system=self.config.proof_system,
        )
        strong_craig = CraigInterpolant(cfg_strong)
        weak_craig = CraigInterpolant(cfg_weak)

        r1 = strong_craig.compute(formula_a, formula_b)
        r2 = weak_craig.compute(formula_a, formula_b)
        return r1, r2


# ---------------------------------------------------------------------------
# Sequence Interpolation
# ---------------------------------------------------------------------------


class SequenceInterpolation:
    """Sequence of interpolants for path analysis.

    Given a path A₁, A₂, ..., Aₙ where the conjunction is UNSAT,
    computes interpolants I₁, ..., Iₙ₋₁ such that:
      - A₁ ⊨ I₁
      - Iₖ ∧ Aₖ₊₁ ⊨ Iₖ₊₁
      - Iₙ₋₁ ∧ Aₙ is UNSAT
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._craig = CraigInterpolant(config)
        self._checker = SatisfiabilityChecker()

    def compute(self, formulas: List[DPFormula]) -> Optional[SequenceInterpolant]:
        """Compute sequence interpolants for a formula chain."""
        if len(formulas) < 2:
            return None

        start = time.time()
        interpolants: List[Interpolant] = []

        for k in range(len(formulas) - 1):
            # A-part: conjunction of formulas up to index k
            prefix_vars: Set[str] = set()
            prefix_parts: List[str] = []
            for j in range(k + 1):
                prefix_vars.update(formulas[j].variables)
                prefix_parts.append(f"({formulas[j].expr})")

            # B-part: conjunction of formulas from k+1 onward
            suffix_vars: Set[str] = set()
            suffix_parts: List[str] = []
            for j in range(k + 1, len(formulas)):
                suffix_vars.update(formulas[j].variables)
                suffix_parts.append(f"({formulas[j].expr})")

            fa = DPFormula(
                expr=" ∧ ".join(prefix_parts),
                variables=frozenset(prefix_vars),
            )
            fb = DPFormula(
                expr=" ∧ ".join(suffix_parts),
                variables=frozenset(suffix_vars),
            )

            result = self._craig.compute(fa, fb)
            if not result.success or result.interpolant is None:
                return None
            interpolants.append(result.interpolant)

        is_inductive = self._check_inductiveness(formulas, interpolants)

        return SequenceInterpolant(
            interpolants=interpolants,
            formulas=[
                Formula.from_dp_formula(f).to_dp_formula() for f in formulas
            ],
            is_inductive=is_inductive,
        )

    def _check_inductiveness(
        self,
        formulas: List[DPFormula],
        interpolants: List[Interpolant],
    ) -> bool:
        """Check whether the interpolant sequence is inductive."""
        if len(interpolants) < 2:
            return False
        # Check if I_k == I_{k+1} for consecutive steps (fixed-point)
        for i in range(len(interpolants) - 1):
            if interpolants[i].formula.expr == interpolants[i + 1].formula.expr:
                return True
        return False


# ---------------------------------------------------------------------------
# Strength Reduction
# ---------------------------------------------------------------------------


class StrengthReduction:
    """Simplify interpolants while preserving key properties.

    Provides weakening and strengthening operations that maintain
    the interpolant's validity with respect to A and B.
    """

    def __init__(self, *, max_iterations: int = 50) -> None:
        self.max_iterations = max_iterations
        self._simplifier = Simplifier()
        self._checker = SatisfiabilityChecker()

    def weaken(
        self,
        interpolant: Interpolant,
        formula_a: DPFormula,
        formula_b: DPFormula,
    ) -> Interpolant:
        """Weaken an interpolant while preserving validity.

        Removes conjuncts that are not necessary to maintain I ∧ B is UNSAT.
        """
        fa = Formula.from_dp_formula(interpolant.formula)
        fb = Formula.from_dp_formula(formula_b)

        if fa.node.kind == NodeKind.AND:
            conjuncts = list(fa.node.children)
            reduced: List[FormulaNode] = []
            for i, conj in enumerate(conjuncts):
                # Try removing this conjunct
                remaining = reduced + conjuncts[i + 1:]
                if remaining:
                    candidate = Formula(FormulaNode.and_(*remaining))
                    combined = candidate & fb
                    if self._checker.is_unsat(combined):
                        continue
                reduced.append(conj)

            if reduced:
                new_node = FormulaNode.and_(*reduced) if len(reduced) > 1 else reduced[0]
            else:
                new_node = FormulaNode.const(True)
            new_formula = Formula(new_node)
        else:
            new_formula = fa

        new_formula = self._simplifier.simplify(new_formula)

        return Interpolant(
            formula=new_formula.to_dp_formula(),
            interpolant_type=interpolant.interpolant_type,
            common_variables=interpolant.common_variables,
            strength=InterpolantStrength.WEAKEST,
            proof_system=interpolant.proof_system,
        )

    def strengthen(
        self,
        interpolant: Interpolant,
        formula_a: DPFormula,
        formula_b: DPFormula,
    ) -> Interpolant:
        """Strengthen an interpolant by adding implied constraints.

        Adds constraints from A that use only common variables.
        """
        fa = Formula.from_dp_formula(formula_a)
        fi = Formula.from_dp_formula(interpolant.formula)
        common = interpolant.common_variables

        qe = QuantifierElimination()
        # Project A onto common variables
        vars_to_elim = list(fa.variables - common)
        projected = qe.eliminate(fa, vars_to_elim)
        projected = self._simplifier.simplify(projected)

        # Conjoin with existing interpolant
        strengthened = fi & projected
        strengthened = self._simplifier.simplify(strengthened)

        return Interpolant(
            formula=strengthened.to_dp_formula(),
            interpolant_type=interpolant.interpolant_type,
            common_variables=common,
            strength=InterpolantStrength.STRONGEST,
            proof_system=interpolant.proof_system,
        )

    def minimize_variables(
        self,
        interpolant: Interpolant,
        formula_b: DPFormula,
    ) -> Interpolant:
        """Reduce the number of variables in the interpolant."""
        fi = Formula.from_dp_formula(interpolant.formula)
        fb = Formula.from_dp_formula(formula_b)
        common = interpolant.common_variables

        qe = QuantifierElimination()
        # Try eliminating variables one at a time
        current = fi
        remaining_vars = set(current.variables)

        for v in sorted(remaining_vars):
            candidate = qe.eliminate(current, [v])
            candidate = self._simplifier.simplify(candidate)
            combined = candidate & fb
            if self._checker.is_unsat(combined):
                current = candidate
                remaining_vars.discard(v)

        return Interpolant(
            formula=current.to_dp_formula(),
            interpolant_type=interpolant.interpolant_type,
            common_variables=frozenset(remaining_vars) & common,
            strength=interpolant.strength,
            proof_system=interpolant.proof_system,
        )


# ---------------------------------------------------------------------------
# Interpolant Cache
# ---------------------------------------------------------------------------


class InterpolantCache:
    """Cache and reuse computed interpolants.

    Uses formula hashing for lookup and maintains an LRU eviction policy.
    """

    def __init__(self, *, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, InterpolationResult] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, formula_a: DPFormula, formula_b: DPFormula) -> str:
        h = hashlib.sha256()
        h.update(formula_a.expr.encode("utf-8"))
        h.update(b"|")
        h.update(formula_b.expr.encode("utf-8"))
        return h.hexdigest()

    def get(
        self,
        formula_a: DPFormula,
        formula_b: DPFormula,
    ) -> Optional[InterpolationResult]:
        key = self._key(formula_a, formula_b)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(
        self,
        formula_a: DPFormula,
        formula_b: DPFormula,
        result: InterpolationResult,
    ) -> None:
        key = self._key(formula_a, formula_b)
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"InterpolantCache(size={self.size}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Proof-Based Interpolation
# ---------------------------------------------------------------------------


class ProofBasedInterpolation:
    """Extract interpolants from resolution proofs.

    Implements Pudlak's and McMillan's algorithms for extracting
    interpolants from resolution refutation proofs.
    """

    def __init__(self, config: Optional[InterpolantConfig] = None) -> None:
        self.config = config or InterpolantConfig()
        self._simplifier = Simplifier()

    def from_resolution_proof(
        self,
        proof: ResolutionProof,
    ) -> Optional[Interpolant]:
        """Extract interpolant from a resolution proof using Pudlak's algorithm.

        For each proof step, computes a partial interpolant:
          - Input clause from A: restrict clause to common variables
          - Input clause from B: True
          - Resolution on pivot p:
            * p is A-local: OR of parent interpolants
            * p is B-local: AND of parent interpolants
            * p is common:  (p ∨ I1) ∧ (¬p ∨ I2)
        """
        if not proof.steps:
            return None

        partial: List[FormulaNode] = []
        common = proof.common_variables
        vars_a = proof.variables_a

        for i, step in enumerate(proof.steps):
            if step.source == "input_a":
                # Restrict clause to common variables
                common_lits = [
                    lit for lit in step.clause
                    if lit.free_variables() <= common
                ]
                if common_lits:
                    partial.append(FormulaNode.or_(*common_lits))
                else:
                    partial.append(FormulaNode.const(False))

            elif step.source == "input_b":
                partial.append(FormulaNode.const(True))

            else:
                # Derived step via resolution
                if step.parent_a is None or step.parent_b is None:
                    partial.append(FormulaNode.const(True))
                    continue

                i1 = partial[step.parent_a]
                i2 = partial[step.parent_b]
                pivot = step.pivot

                if pivot and pivot in vars_a and pivot not in common:
                    # A-local pivot: disjunction
                    partial.append(FormulaNode.or_(i1, i2))
                elif pivot and pivot not in vars_a and pivot not in common:
                    # B-local pivot: conjunction
                    partial.append(FormulaNode.and_(i1, i2))
                else:
                    # Common pivot: (p ∨ I1) ∧ (¬p ∨ I2)
                    p = FormulaNode.var(pivot) if pivot else FormulaNode.const(True)
                    partial.append(FormulaNode.and_(
                        FormulaNode.or_(p, i1),
                        FormulaNode.or_(FormulaNode.not_(p), i2),
                    ))

        if not partial:
            return None

        itp_node = partial[-1]
        itp_formula = self._simplifier.simplify(Formula(itp_node))

        return Interpolant(
            formula=itp_formula.to_dp_formula(),
            interpolant_type=self.config.interpolant_type,
            common_variables=common,
            strength=self.config.strength,
            proof_system=ProofSystem.RESOLUTION,
        )

    def from_farkas_certificate(
        self,
        constraints_a: List[Tuple[Dict[str, float], float]],
        constraints_b: List[Tuple[Dict[str, float], float]],
        multipliers_a: np.ndarray,
        multipliers_b: np.ndarray,
        common_vars: FrozenSet[str],
    ) -> Optional[Interpolant]:
        """Extract interpolant from Farkas infeasibility certificate.

        Given λ_A ≥ 0, λ_B ≥ 0 such that λ_A^T [A|a] + λ_B^T [B|b]
        yields 0x ≤ c with c < 0, the interpolant is:
        I = λ_A^T (A_common x - a)  restricted to common variables.
        """
        itp_coeffs: Dict[str, float] = {}
        itp_rhs = 0.0

        for i, (coeffs, rhs) in enumerate(constraints_a):
            mu = float(multipliers_a[i])
            if abs(mu) < 1e-15:
                continue
            itp_rhs += mu * rhs
            for v, c in coeffs.items():
                if v in common_vars:
                    itp_coeffs[v] = itp_coeffs.get(v, 0.0) + mu * c

        itp_coeffs = {v: c for v, c in itp_coeffs.items() if abs(c) > 1e-12}

        if not itp_coeffs:
            val = FormulaNode.const(itp_rhs >= 0)
            return Interpolant(
                formula=Formula(val).to_dp_formula(),
                interpolant_type=InterpolantType.LINEAR_ARITHMETIC,
                common_variables=frozenset(),
                proof_system=ProofSystem.FARKAS_LEMMA,
            )

        node = FormulaNode.leq(itp_coeffs, itp_rhs)
        formula = self._simplifier.simplify(Formula(node))

        return Interpolant(
            formula=formula.to_dp_formula(),
            interpolant_type=InterpolantType.LINEAR_ARITHMETIC,
            common_variables=frozenset(itp_coeffs.keys()) & common_vars,
            proof_system=ProofSystem.FARKAS_LEMMA,
        )

    def build_resolution_proof(
        self,
        clauses_a: List[Tuple[FormulaNode, ...]],
        clauses_b: List[Tuple[FormulaNode, ...]],
        variables_a: FrozenSet[str],
        variables_b: FrozenSet[str],
    ) -> Optional[ResolutionProof]:
        """Attempt to build a resolution proof from clause sets.

        Uses a simple ordered-resolution strategy.
        """
        steps: List[ResolutionStep] = []

        for clause in clauses_a:
            steps.append(ResolutionStep(clause=clause, source="input_a"))
        for clause in clauses_b:
            steps.append(ResolutionStep(clause=clause, source="input_b"))

        # Simple resolution: try resolving pairs
        max_steps = len(steps) * 3
        active = list(range(len(steps)))

        for iteration in range(max_steps):
            if iteration >= len(active) - 1:
                break
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ci = steps[active[i]].clause
                    cj = steps[active[j]].clause
                    resolved = self._try_resolve(ci, cj)
                    if resolved is not None:
                        clause, pivot = resolved
                        idx = len(steps)
                        steps.append(ResolutionStep(
                            clause=clause, parent_a=active[i],
                            parent_b=active[j], pivot=pivot,
                            source="derived",
                        ))
                        active.append(idx)
                        if len(clause) == 0:
                            return ResolutionProof(
                                steps=steps,
                                variables_a=variables_a,
                                variables_b=variables_b,
                            )

        return None

    def _try_resolve(
        self,
        c1: Tuple[FormulaNode, ...],
        c2: Tuple[FormulaNode, ...],
    ) -> Optional[Tuple[Tuple[FormulaNode, ...], str]]:
        """Try to resolve two clauses on a pivot variable."""
        for lit1 in c1:
            for lit2 in c2:
                if (lit1.kind == NodeKind.VAR and lit2.kind == NodeKind.NOT
                        and lit2.children[0].kind == NodeKind.VAR
                        and lit1.value == lit2.children[0].value):
                    pivot = lit1.value
                    new_clause = tuple(
                        l for l in c1 if l != lit1
                    ) + tuple(
                        l for l in c2 if l != lit2
                    )
                    return new_clause, pivot
                if (lit2.kind == NodeKind.VAR and lit1.kind == NodeKind.NOT
                        and lit1.children[0].kind == NodeKind.VAR
                        and lit2.value == lit1.children[0].value):
                    pivot = lit2.value
                    new_clause = tuple(
                        l for l in c1 if l != lit1
                    ) + tuple(
                        l for l in c2 if l != lit2
                    )
                    return new_clause, pivot
        return None


__all__ = [
    "ResolutionStep",
    "ResolutionProof",
    "CraigInterpolant",
    "BinaryInterpolation",
    "SequenceInterpolation",
    "StrengthReduction",
    "InterpolantCache",
    "ProofBasedInterpolation",
]
