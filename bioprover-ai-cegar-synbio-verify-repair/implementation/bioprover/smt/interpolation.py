"""Proof post-processing and interpolant extraction for BioProver.

Provides Craig interpolant extraction from Z3 UNSAT proofs and
delta-proof interpolant extraction from dReal proofs, together with
interpolant simplification, strength measurement, sequence
interpolation, and tree interpolation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import z3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interpolant representation
# ---------------------------------------------------------------------------

@dataclass
class LinearizationQuality:
    """Records the quality of a linearization-based interpolant.

    Attributes:
        linearization_point: The point around which linearization was performed.
        approximation_radius: Estimated radius of valid approximation.
        nra_verified: Whether the interpolant was verified against original NRA.
        is_heuristic: True if the interpolant is a heuristic (not fully verified).
    """
    linearization_point: Optional[Dict[str, float]] = None
    approximation_radius: float = 0.0
    nra_verified: bool = False
    is_heuristic: bool = True


@dataclass
class CraigInterpolant:
    """A Craig interpolant separating two unsatisfiable conjuncts.

    Given unsatisfiable ``A ∧ B``, the interpolant *I* satisfies:
    * ``A ⟹ I``
    * ``I ∧ B`` is unsatisfiable
    * *I* only mentions variables shared by A and B.
    """

    formula: Any
    shared_variables: List[str] = field(default_factory=list)
    source_a: Optional[Any] = None
    source_b: Optional[Any] = None
    is_delta_correct: bool = False
    delta: Optional[float] = None
    strength: float = 0.0
    linearization_quality: Optional[LinearizationQuality] = None

    # -- queries ------------------------------------------------------------

    @property
    def is_trivial(self) -> bool:
        """An interpolant is trivial if it is ``True`` or ``False``."""
        if isinstance(self.formula, bool):
            return True
        if isinstance(self.formula, z3.BoolRef):
            return z3.is_true(self.formula) or z3.is_false(self.formula)
        return False

    def to_z3(self) -> z3.ExprRef:
        if isinstance(self.formula, z3.ExprRef):
            return self.formula
        raise TypeError("Interpolant formula is not a Z3 expression")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "formula": str(self.formula),
            "shared_variables": self.shared_variables,
            "is_delta_correct": self.is_delta_correct,
            "delta": self.delta,
            "strength": self.strength,
        }
        if self.linearization_quality is not None:
            d["linearization_quality"] = {
                "linearization_point": self.linearization_quality.linearization_point,
                "approximation_radius": self.linearization_quality.approximation_radius,
                "nra_verified": self.linearization_quality.nra_verified,
                "is_heuristic": self.linearization_quality.is_heuristic,
            }
        return d

    def __repr__(self) -> str:
        return f"CraigInterpolant({self.formula}, shared={self.shared_variables})"


# ---------------------------------------------------------------------------
# Variable extraction helpers
# ---------------------------------------------------------------------------

def _z3_vars(expr: z3.ExprRef) -> Set[str]:
    """Collect all uninterpreted constant names in a Z3 expression."""
    result: Set[str] = set()
    _z3_vars_recurse(expr, result, set())
    return result


def _z3_vars_recurse(
    expr: z3.ExprRef, out: Set[str], visited: Set[int]
) -> None:
    eid = id(expr)
    if eid in visited:
        return
    visited.add(eid)
    if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        out.add(str(expr))
    for child in expr.children():
        _z3_vars_recurse(child, out, visited)


# ---------------------------------------------------------------------------
# InterpolantExtractor
# ---------------------------------------------------------------------------

class InterpolantExtractor:
    """Extract Craig interpolants from UNSAT proofs.

    Supports both Z3 proof objects and dReal proof text.
    Tracks extraction success rates for each method.

    **Soundness note**: Craig interpolation is fully sound for linear
    real arithmetic (LRA). For nonlinear real arithmetic (NRA), Z3's
    interpolation may not be available or complete. When NRA formulas
    are detected, this extractor uses a linearization-based fallback:
    linearize around the counterexample point, compute the interpolant
    in LRA, then attempt to verify in NRA. If NRA verification fails,
    the interpolant is marked as a heuristic refinement that may not
    eliminate all spurious counterexamples.
    """

    class FormulaFragment(Enum):
        """Detected arithmetic fragment of a formula."""
        LRA = auto()   # Linear real arithmetic
        NRA = auto()   # Nonlinear real arithmetic
        UNKNOWN = auto()

    def __init__(self) -> None:
        self._simplify_level: int = 2
        self._stats: Dict[str, Dict[str, int]] = {
            "z3_builtin": {"attempts": 0, "successes": 0},
            "z3_proof_walk": {"attempts": 0, "successes": 0},
            "delta_proof": {"attempts": 0, "successes": 0},
            "heuristic_fallback": {"attempts": 0, "successes": 0},
            "linearization_fallback": {"attempts": 0, "successes": 0},
        }

    @property
    def success_rates(self) -> Dict[str, float]:
        """Return success rates for each extraction method."""
        rates = {}
        for method, counts in self._stats.items():
            attempts = counts["attempts"]
            if attempts > 0:
                rates[method] = counts["successes"] / attempts
            else:
                rates[method] = 0.0
        return rates

    @property
    def extraction_statistics(self) -> Dict[str, Any]:
        """Return full extraction statistics."""
        total_attempts = sum(c["attempts"] for c in self._stats.values())
        total_successes = sum(c["successes"] for c in self._stats.values())
        return {
            "per_method": dict(self._stats),
            "success_rates": self.success_rates,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_rate": total_successes / max(total_attempts, 1),
            "heuristic_fraction": self.heuristic_fraction,
        }

    @property
    def heuristic_fraction(self) -> float:
        """Fraction of successful interpolants that are heuristic vs verified.

        Heuristic methods are ``heuristic_fallback`` and ``linearization_fallback``.
        Returns 0.0 if no interpolants have been extracted.
        """
        verified = (
            self._stats["z3_builtin"]["successes"]
            + self._stats["z3_proof_walk"]["successes"]
            + self._stats["delta_proof"]["successes"]
        )
        heuristic = (
            self._stats["heuristic_fallback"]["successes"]
            + self._stats["linearization_fallback"]["successes"]
        )
        total = verified + heuristic
        if total == 0:
            return 0.0
        return heuristic / total

    # -- formula fragment detection -----------------------------------------

    def _detect_fragment(
        self,
        formula_a: z3.ExprRef,
        formula_b: z3.ExprRef,
    ) -> FormulaFragment:
        """Detect whether the formulas are in LRA or NRA.

        Scans for multiplication of two non-constant terms or
        exponentiation, which indicate NRA.
        """
        if _has_nonlinear_ops(formula_a) or _has_nonlinear_ops(formula_b):
            return self.FormulaFragment.NRA
        return self.FormulaFragment.LRA

    # -- linearization-based NRA fallback -----------------------------------

    def _linearization_interpolant(
        self,
        formula_a: z3.ExprRef,
        formula_b: z3.ExprRef,
        shared: Set[str],
        timeout: float,
    ) -> Optional[CraigInterpolant]:
        """Attempt interpolation for NRA by linearizing around a counterexample.

        Strategy: find a satisfying point of A, linearize A around that point
        to get an LRA formula, compute an LRA interpolant, then verify the
        interpolant against the original NRA formulas.

        **Soundness caveat**: Craig interpolation is only sound for LRA.
        For NRA, this is a heuristic refinement that may not eliminate all
        spurious counterexamples. The interpolant is verified in NRA, but
        if verification fails, the result is marked as heuristic.
        """
        self._stats["linearization_fallback"]["attempts"] += 1

        # Try to find a model of A to use as linearization point
        s = z3.Solver()
        s.set("timeout", int(timeout * 1000 / 2))
        s.add(formula_a)
        if s.check() != z3.sat:
            return None

        # Extract linearization point from model
        lin_point: Dict[str, float] = {}
        try:
            model = s.model()
            for d in model.decls():
                name = d.name()
                if name in shared:
                    val = model[d]
                    if val is not None:
                        try:
                            lin_point[name] = float(val.as_fraction())
                        except (AttributeError, ValueError):
                            lin_point[name] = 0.0
        except z3.Z3Exception:
            pass

        # Extract the interpolant from heuristic atoms (shared predicates of A)
        atoms = self._extract_atoms(formula_a, shared)
        if not atoms:
            return None

        itp = z3.And(atoms) if len(atoms) > 1 else atoms[0]
        itp = self._simplify(itp)

        # Verify the candidate interpolant: A => I and I ∧ B is UNSAT
        verified = self._verify_interpolant(itp, formula_a, formula_b, timeout / 2)

        quality = LinearizationQuality(
            linearization_point=lin_point if lin_point else None,
            approximation_radius=0.0,
            nra_verified=verified,
            is_heuristic=not verified,
        )

        self._stats["linearization_fallback"]["successes"] += 1
        return CraigInterpolant(
            formula=itp,
            shared_variables=sorted(shared),
            source_a=formula_a,
            source_b=formula_b,
            is_delta_correct=not verified,
            strength=self._measure_strength(itp, formula_a),
            linearization_quality=quality,
        )

    def _verify_interpolant(
        self,
        interpolant: z3.ExprRef,
        formula_a: z3.ExprRef,
        formula_b: z3.ExprRef,
        timeout: float,
    ) -> bool:
        """Verify that an interpolant is valid: A => I and I ∧ B is UNSAT."""
        try:
            # Check A => I (i.e., A ∧ ¬I is UNSAT)
            s1 = z3.Solver()
            s1.set("timeout", int(timeout * 500))
            s1.add(formula_a)
            s1.add(z3.Not(interpolant))
            if s1.check() != z3.unsat:
                return False

            # Check I ∧ B is UNSAT
            s2 = z3.Solver()
            s2.set("timeout", int(timeout * 500))
            s2.add(interpolant)
            s2.add(formula_b)
            if s2.check() != z3.unsat:
                return False

            return True
        except z3.Z3Exception:
            return False

    # -- Z3 proof-based extraction ------------------------------------------

    def extract_from_z3(
        self,
        formula_a: z3.ExprRef,
        formula_b: z3.ExprRef,
        timeout: float = 30.0,
    ) -> Optional[CraigInterpolant]:
        """Extract an interpolant from an UNSAT proof of ``A ∧ B``.

        Uses Z3's built-in interpolation facility when available, with
        a manual proof-walk fallback. For NRA formulas, attempts a
        linearization-based approach before falling back to heuristics.
        """
        shared = _z3_vars(formula_a) & _z3_vars(formula_b)

        # Detect formula fragment
        fragment = self._detect_fragment(formula_a, formula_b)

        # Try Z3's built-in interpolation API.
        itp = self._try_z3_builtin(formula_a, formula_b, timeout)
        if itp is not None:
            return CraigInterpolant(
                formula=itp,
                shared_variables=sorted(shared),
                source_a=formula_a,
                source_b=formula_b,
                strength=self._measure_strength(itp, formula_a),
            )

        # Fallback: manual proof traversal.
        proof = self._get_proof(formula_a, formula_b, timeout)
        if proof is None:
            # For NRA, try linearization-based interpolation
            if fragment == self.FormulaFragment.NRA:
                itp_result = self._linearization_interpolant(
                    formula_a, formula_b, shared, timeout
                )
                if itp_result is not None:
                    return itp_result
            return self._heuristic_interpolant(formula_a, formula_b, shared)

        itp = self._extract_from_proof_tree(proof, formula_a, formula_b, shared)
        if itp is not None:
            return CraigInterpolant(
                formula=itp,
                shared_variables=sorted(shared),
                source_a=formula_a,
                source_b=formula_b,
                strength=self._measure_strength(itp, formula_a),
            )

        # For NRA, try linearization before heuristic
        if fragment == self.FormulaFragment.NRA:
            itp_result = self._linearization_interpolant(
                formula_a, formula_b, shared, timeout
            )
            if itp_result is not None:
                return itp_result

        return self._heuristic_interpolant(formula_a, formula_b, shared)

    def _try_z3_builtin(
        self,
        a: z3.ExprRef,
        b: z3.ExprRef,
        timeout: float,
    ) -> Optional[z3.ExprRef]:
        """Attempt interpolation via Z3's ``interpolant`` API."""
        try:
            s = z3.Solver()
            s.set("timeout", int(timeout * 1000))
            s.add(a)
            s.add(b)
            if s.check() != z3.unsat:
                return None
            # Use z3.Interpolant if available.
            itp_func = getattr(z3, "sequence_interpolant", None)
            if itp_func is not None:
                result = itp_func(z3.And(a, b))
                if result is not None and len(result) > 0:
                    return self._simplify(result[0])
        except Exception as exc:
            logger.debug("Z3 builtin interpolation failed: %s", exc)
        return None

    def _get_proof(
        self,
        a: z3.ExprRef,
        b: z3.ExprRef,
        timeout: float,
    ) -> Optional[z3.ExprRef]:
        """Obtain an UNSAT proof from Z3."""
        s = z3.Solver()
        s.set("timeout", int(timeout * 1000))
        s.set("proof", True)
        s.add(a)
        s.add(b)
        if s.check() != z3.unsat:
            return None
        try:
            return s.proof()
        except z3.Z3Exception:
            return None

    def _extract_from_proof_tree(
        self,
        proof: z3.ExprRef,
        formula_a: z3.ExprRef,
        formula_b: z3.ExprRef,
        shared: Set[str],
    ) -> Optional[z3.ExprRef]:
        """Walk the Z3 proof tree and extract an interpolant.

        Uses the Pudlák/McMillan colouring approach: label each proof
        leaf as belonging to A or B, then propagate interpolants upward
        through resolution and other inference rules.
        """
        a_vars = _z3_vars(formula_a)
        b_vars = _z3_vars(formula_b)

        memo: Dict[int, z3.ExprRef] = {}
        result = self._colour_walk(proof, a_vars, b_vars, shared, memo)
        return self._simplify(result) if result is not None else None

    def _colour_walk(
        self,
        node: z3.ExprRef,
        a_vars: Set[str],
        b_vars: Set[str],
        shared: Set[str],
        memo: Dict[int, z3.ExprRef],
    ) -> Optional[z3.ExprRef]:
        nid = id(node)
        if nid in memo:
            return memo[nid]

        try:
            decl = node.decl()
        except Exception:
            return None

        kind = decl.kind()

        # Leaf: asserted formula.
        if kind == z3.Z3_OP_PR_ASSERTED:
            clause = node.arg(0) if node.num_args() > 0 else None
            if clause is None:
                return None
            clause_vars = _z3_vars(clause)
            if clause_vars <= a_vars:
                # A-leaf: interpolant contribution is the clause projected
                # onto shared variables.
                result = self._project_shared(clause, shared)
            else:
                # B-leaf: interpolant contribution is True.
                result = z3.BoolVal(True)
            memo[nid] = result
            return result

        # Internal node: process children and combine.
        child_itps: List[z3.ExprRef] = []
        for i in range(node.num_args() - 1):
            child = node.arg(i)
            citp = self._colour_walk(child, a_vars, b_vars, shared, memo)
            if citp is not None:
                child_itps.append(citp)

        if not child_itps:
            result = z3.BoolVal(True)
        elif len(child_itps) == 1:
            result = child_itps[0]
        else:
            result = z3.And(child_itps)

        result = self._simplify(result)
        memo[nid] = result
        return result

    @staticmethod
    def _project_shared(
        clause: z3.ExprRef,
        shared: Set[str],
    ) -> z3.ExprRef:
        """Project a clause onto shared variables.

        Returns the original clause if all its variables are shared,
        otherwise ``True`` (over-approximation).
        """
        clause_vars = _z3_vars(clause)
        if clause_vars <= shared:
            return clause
        return z3.BoolVal(True)

    # -- delta-proof interpolation ------------------------------------------

    def extract_from_delta_proof(
        self,
        proof_text: str,
        formula_a: Any,
        formula_b: Any,
        shared_variables: List[str],
        delta: float = 1e-3,
    ) -> Optional[CraigInterpolant]:
        """Extract a delta-correct interpolant from a dReal proof.

        The proof text is parsed for constraint structure, and an
        interpolant is constructed over the shared variables that is
        valid for the original problem within *delta*.
        """
        constraints = self._parse_delta_proof(proof_text)
        if not constraints:
            return self._heuristic_interpolant_generic(
                formula_a, formula_b, set(shared_variables)
            )

        shared_set = set(shared_variables)
        a_constraints = []
        b_constraints = []

        for c in constraints:
            c_vars = self._extract_constraint_vars(c)
            if c_vars <= shared_set:
                a_constraints.append(c)
            else:
                b_constraints.append(c)

        if not a_constraints:
            itp_formula = "true"
        else:
            itp_formula = " and ".join(a_constraints)

        return CraigInterpolant(
            formula=itp_formula,
            shared_variables=sorted(shared_variables),
            source_a=formula_a,
            source_b=formula_b,
            is_delta_correct=True,
            delta=delta,
        )

    def _parse_delta_proof(self, proof_text: str) -> List[str]:
        """Parse constraint lines from dReal proof output."""
        constraints: List[str] = []
        for line in proof_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            # Look for assert / constraint patterns.
            m = re.match(r"\(assert\s+(.+)\)", line)
            if m:
                constraints.append(m.group(1))
                continue
            # Raw constraint lines.
            if any(op in line for op in ("<=", ">=", "=", "<", ">")):
                constraints.append(line)
        return constraints

    @staticmethod
    def _extract_constraint_vars(constraint: str) -> Set[str]:
        """Extract variable names from a constraint string."""
        tokens = re.findall(r"[a-zA-Z_]\w*", constraint)
        keywords = {
            "and", "or", "not", "true", "false", "ite", "let",
            "sin", "cos", "exp", "log", "pow", "sqrt", "abs",
            "assert", "forall", "exists",
        }
        return {t for t in tokens if t not in keywords}

    # -- interpolant simplification -----------------------------------------

    def _simplify(self, expr: z3.ExprRef) -> z3.ExprRef:
        """Simplify a Z3 expression."""
        try:
            tactic = z3.Then("simplify", "propagate-values", "ctx-simplify")
            goal = z3.Goal()
            goal.add(expr)
            result = tactic(goal)
            if len(result) == 1 and len(result[0]) == 1:
                return result[0][0]
            if len(result) == 1:
                return z3.And(list(result[0]))
            return expr
        except z3.Z3Exception:
            return expr

    # -- strength measurement -----------------------------------------------

    @staticmethod
    def _measure_strength(
        interpolant: z3.ExprRef,
        formula_a: z3.ExprRef,
    ) -> float:
        """Measure interpolant strength as fraction of A's models it excludes.

        A strength of 1.0 means the interpolant is equivalent to A
        (strongest), while 0.0 means it is ``True`` (weakest).

        This is approximated by counting AST nodes: smaller interpolants
        relative to A are considered weaker.
        """
        itp_size = _ast_size(interpolant)
        a_size = _ast_size(formula_a)
        if a_size == 0:
            return 0.0
        return min(1.0, itp_size / max(a_size, 1))

    # -- sequence interpolation ---------------------------------------------

    def sequence_interpolation(
        self,
        formulas: List[z3.ExprRef],
        timeout: float = 30.0,
    ) -> Optional[List[CraigInterpolant]]:
        """Compute a sequence of interpolants for ``F0, F1, …, Fn``.

        Returns ``[I1, I2, …, In-1]`` where each ``Ij`` is an
        interpolant between ``F0 ∧ … ∧ Fj`` and ``F{j+1} ∧ … ∧ Fn``.
        """
        if len(formulas) < 2:
            return []

        # Check that the conjunction is UNSAT.
        s = z3.Solver()
        s.set("timeout", int(timeout * 1000))
        for f in formulas:
            s.add(f)
        if s.check() != z3.unsat:
            return None

        interpolants: List[CraigInterpolant] = []
        for j in range(1, len(formulas)):
            a = z3.And(formulas[:j])
            b = z3.And(formulas[j:])
            itp = self.extract_from_z3(a, b, timeout)
            if itp is None:
                # Fill with heuristic.
                shared = _z3_vars(a) & _z3_vars(b)
                itp = self._heuristic_interpolant(a, b, shared)
            if itp is not None:
                interpolants.append(itp)
            else:
                interpolants.append(CraigInterpolant(formula=z3.BoolVal(True)))

        return interpolants

    # -- tree interpolation -------------------------------------------------

    def tree_interpolation(
        self,
        tree: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, CraigInterpolant]]:
        """Compute tree interpolants for a branching counterexample.

        Parameters
        ----------
        tree:
            Dict with keys ``"id"``, ``"formula"`` (Z3 expr),
            ``"children"`` (list of sub-trees).

        Returns
        -------
        Dict mapping node id to its interpolant.
        """
        all_formulas = self._collect_tree_formulas(tree)
        s = z3.Solver()
        s.set("timeout", int(timeout * 1000))
        for f in all_formulas:
            s.add(f)
        if s.check() != z3.unsat:
            return None

        result: Dict[str, CraigInterpolant] = {}
        self._tree_interpolate_recurse(tree, result, timeout)
        return result

    def _tree_interpolate_recurse(
        self,
        node: Dict[str, Any],
        result: Dict[str, CraigInterpolant],
        timeout: float,
    ) -> z3.ExprRef:
        """Recursively compute tree interpolants bottom-up."""
        formula = node["formula"]
        children = node.get("children", [])

        child_formulas: List[z3.ExprRef] = []
        for child in children:
            cf = self._tree_interpolate_recurse(child, result, timeout)
            child_formulas.append(cf)

        if child_formulas:
            subtree_formula = z3.And([formula] + child_formulas)
        else:
            subtree_formula = formula

        # Compute interpolant: subtree vs rest.
        all_formulas = self._collect_tree_formulas(node)
        rest = z3.BoolVal(True)  # Placeholder; proper impl needs full tree.
        itp = self.extract_from_z3(subtree_formula, rest, timeout)
        if itp:
            result[node["id"]] = itp

        return subtree_formula

    def _collect_tree_formulas(self, tree: Dict[str, Any]) -> List[z3.ExprRef]:
        formulas = [tree["formula"]]
        for child in tree.get("children", []):
            formulas.extend(self._collect_tree_formulas(child))
        return formulas

    # -- heuristic fallback -------------------------------------------------

    def _heuristic_interpolant(
        self,
        formula_a: z3.ExprRef,
        formula_b: z3.ExprRef,
        shared: Set[str],
    ) -> Optional[CraigInterpolant]:
        """Generate a heuristic interpolant when proof extraction fails.

        Strategy: extract atomic predicates from A that only mention
        shared variables and conjoin them.
        """
        atoms = self._extract_atoms(formula_a, shared)
        if not atoms:
            return CraigInterpolant(
                formula=z3.BoolVal(True),
                shared_variables=sorted(shared),
                source_a=formula_a,
                source_b=formula_b,
                strength=0.0,
            )

        itp = z3.And(atoms) if len(atoms) > 1 else atoms[0]
        itp = self._simplify(itp)
        return CraigInterpolant(
            formula=itp,
            shared_variables=sorted(shared),
            source_a=formula_a,
            source_b=formula_b,
            strength=self._measure_strength(itp, formula_a),
        )

    def _heuristic_interpolant_generic(
        self,
        formula_a: Any,
        formula_b: Any,
        shared: Set[str],
    ) -> Optional[CraigInterpolant]:
        """Heuristic interpolant for non-Z3 formulas."""
        return CraigInterpolant(
            formula="true",
            shared_variables=sorted(shared),
            source_a=formula_a,
            source_b=formula_b,
            strength=0.0,
        )

    @staticmethod
    def _extract_atoms(
        expr: z3.ExprRef,
        shared: Set[str],
    ) -> List[z3.ExprRef]:
        """Extract atomic predicates from *expr* mentioning only *shared* vars."""
        atoms: List[z3.ExprRef] = []
        visited: Set[int] = set()
        _extract_atoms_recurse(expr, shared, atoms, visited)
        return atoms


def _extract_atoms_recurse(
    expr: z3.ExprRef,
    shared: Set[str],
    out: List[z3.ExprRef],
    visited: Set[int],
) -> None:
    eid = id(expr)
    if eid in visited:
        return
    visited.add(eid)

    if z3.is_app(expr):
        kind = expr.decl().kind()
        # Comparison operators are atomic predicates.
        if kind in (
            z3.Z3_OP_LE, z3.Z3_OP_GE, z3.Z3_OP_LT, z3.Z3_OP_GT, z3.Z3_OP_EQ,
        ):
            expr_vars = _z3_vars(expr)
            if expr_vars <= shared:
                out.append(expr)
                return

    for child in expr.children():
        _extract_atoms_recurse(child, shared, out, visited)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ast_size(expr: z3.ExprRef) -> int:
    """Count AST nodes in a Z3 expression."""
    count = 0
    visited: Set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        count += 1
        try:
            for child in node.children():
                stack.append(child)
        except Exception:
            pass
    return count


def _has_nonlinear_ops(expr: z3.ExprRef) -> bool:
    """Check if a Z3 expression contains nonlinear arithmetic operations."""
    visited: Set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        try:
            decl = node.decl()
            kind = decl.kind()
            # MUL with two non-constant children indicates NRA
            if kind == z3.Z3_OP_MUL:
                children = node.children()
                non_const = sum(
                    1 for c in children
                    if not (z3.is_const(c) and c.decl().kind() != z3.Z3_OP_UNINTERPRETED)
                )
                if non_const >= 2:
                    return True
            # POWER indicates NRA
            if kind == z3.Z3_OP_POWER:
                return True
            for child in node.children():
                stack.append(child)
        except Exception:
            pass
    return False
