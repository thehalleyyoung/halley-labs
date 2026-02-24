"""
SMT Encoding & Z3 Interface for Refinement Type Inference.

Theory: QF_UFLIA extended with finite-domain Tag sort.
Decidable fragment – coNP-complete for refinement subtyping.

Encodes refinement predicates, subtyping queries, counterexample
paths, and interpolant extraction using the Z3 theorem prover.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import z3

from src.types import (
    AndPred,
    BaseType,
    BinOpExpr,
    BoolType,
    ComparisonPred,
    ConstExpr,
    FalsePred,
    HasAttrPred,
    IntType,
    LenExpr,
    LinearExpr,
    ListType,
    NoneType_,
    NotPred,
    NullityPred,
    OrPred,
    Predicate,
    RefinementType,
    ScaleExpr,
    StrType,
    TruePred,
    TruthinessPred,
    TypeTagPred,
    VarExpr,
)

# ---------------------------------------------------------------------------
# Canonical tag names for the finite-domain Tag sort
# ---------------------------------------------------------------------------

_DEFAULT_TAGS: Tuple[str, ...] = (
    "int",
    "str",
    "bool",
    "NoneType",
    "list",
    "tuple",
    "dict",
    "set",
    "float",
    "bytes",
    "object",
)

# ---------------------------------------------------------------------------
# SMTResult – outcome of an SMT query
# ---------------------------------------------------------------------------


class Satisfiability(Enum):
    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class SMTResult:
    """Result of an SMT solver invocation."""

    status: Satisfiability
    model: Optional[Dict[str, Any]] = None
    proof: Optional[z3.ExprRef] = None
    unsat_core: Optional[List[z3.ExprRef]] = None
    time_taken: float = 0.0

    @property
    def is_sat(self) -> bool:
        return self.status is Satisfiability.SAT

    @property
    def is_unsat(self) -> bool:
        return self.status is Satisfiability.UNSAT

    @property
    def is_unknown(self) -> bool:
        return self.status is Satisfiability.UNKNOWN


# ---------------------------------------------------------------------------
# SMTContext – manages Z3 context, sorts, and function declarations
# ---------------------------------------------------------------------------

_smt_context_counter = itertools.count()


class SMTContext:
    """Central Z3 context holding sorts, function symbols, and variable map.

    Creates:
      - Tag: finite-domain EnumSort for runtime type tags
      - Int, Bool: standard Z3 sorts
      - Uninterpreted functions: len, isinstance, is_none, is_truthy, hasattr
    """

    def __init__(
        self,
        tag_names: Sequence[str] = _DEFAULT_TAGS,
        *,
        ctx: Optional[z3.Context] = None,
    ) -> None:
        self.ctx = ctx

        # -- sorts -----------------------------------------------------------
        # Use a unique name per context to avoid Z3 redeclaration errors
        _tag_id = next(_smt_context_counter)
        tag_sort_name = f"Tag_{_tag_id}" if _tag_id > 0 else "Tag"
        self.tag_sort, self.tag_values = z3.EnumSort(
            tag_sort_name, list(tag_names), ctx=self.ctx
        )
        self._tag_by_name: Dict[str, z3.ExprRef] = {
            str(v): v for v in self.tag_values
        }
        self.int_sort = z3.IntSort(ctx=self.ctx)
        self.bool_sort = z3.BoolSort(ctx=self.ctx)

        # -- uninterpreted functions -----------------------------------------
        self.len_fn = z3.Function(
            "len", self.int_sort, self.int_sort
        )
        self.isinstance_fn = z3.Function(
            "isinstance", self.int_sort, self.tag_sort, self.bool_sort
        )
        self.is_none_fn = z3.Function(
            "is_none", self.int_sort, self.bool_sort
        )
        self.is_truthy_fn = z3.Function(
            "is_truthy", self.int_sort, self.bool_sort
        )
        self.hasattr_fn = z3.Function(
            "hasattr_check", self.int_sort, self.int_sort, self.bool_sort
        )

        # -- tag(x) returns the tag of a value --------------------------------
        self.tag_fn = z3.Function(
            "tag", self.int_sort, self.tag_sort
        )

        # -- variable map  var_name -> Z3 const -------------------------------
        self._vars: Dict[str, z3.ExprRef] = {}

        # -- attribute string interning map -----------------------------------
        self._attr_consts: Dict[str, z3.ArithRef] = {}
        self._attr_counter: int = 0

        # -- solver stack depth -----------------------------------------------
        self._stack_depth: int = 0

    # -- variable management -------------------------------------------------

    def get_var(self, name: str) -> z3.ArithRef:
        """Return the Z3 Int constant for *name*, creating it lazily."""
        if name not in self._vars:
            self._vars[name] = z3.Int(name, ctx=self.ctx)
        return self._vars[name]  # type: ignore[return-value]

    def get_bool_var(self, name: str) -> z3.BoolRef:
        """Return a Z3 Bool constant for *name*."""
        key = f"__bool__{name}"
        if key not in self._vars:
            self._vars[key] = z3.Bool(name, ctx=self.ctx)
        return self._vars[key]  # type: ignore[return-value]

    def get_tag_var(self, name: str) -> z3.ExprRef:
        """Return a Z3 Tag-sorted constant for *name*."""
        key = f"__tag__{name}"
        if key not in self._vars:
            self._vars[key] = z3.Const(name + "_tag", self.tag_sort)
        return self._vars[key]

    def tag_const(self, tag_name: str) -> z3.ExprRef:
        """Return the Tag enum constant for *tag_name*."""
        if tag_name not in self._tag_by_name:
            raise ValueError(
                f"Unknown tag {tag_name!r}. Known: {list(self._tag_by_name)}"
            )
        return self._tag_by_name[tag_name]

    def has_tag(self, tag_name: str) -> bool:
        return tag_name in self._tag_by_name

    def intern_attr(self, attr: str) -> z3.ArithRef:
        """Intern an attribute name as an integer constant."""
        if attr not in self._attr_consts:
            self._attr_consts[attr] = z3.IntVal(
                self._attr_counter, ctx=self.ctx
            )
            self._attr_counter += 1
        return self._attr_consts[attr]

    def fresh_var(self, prefix: str = "_z") -> z3.ArithRef:
        """Create a fresh integer variable."""
        name = f"{prefix}!{len(self._vars)}"
        return self.get_var(name)

    def fresh_bool(self, prefix: str = "_b") -> z3.BoolRef:
        name = f"{prefix}!{len(self._vars)}"
        return self.get_bool_var(name)

    # -- push / pop ----------------------------------------------------------

    def push(self, solver: z3.Solver) -> None:
        solver.push()
        self._stack_depth += 1

    def pop(self, solver: z3.Solver, n: int = 1) -> None:
        solver.pop(n)
        self._stack_depth = max(0, self._stack_depth - n)

    @property
    def stack_depth(self) -> int:
        return self._stack_depth

    def reset(self) -> None:
        """Clear the variable map (does not affect Z3 context)."""
        self._vars.clear()
        self._attr_consts.clear()
        self._attr_counter = 0
        self._stack_depth = 0

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SMTContext(tags={list(self._tag_by_name)}, "
            f"vars={list(self._vars)}, depth={self._stack_depth})"
        )


# ---------------------------------------------------------------------------
# SMTEncoder – predicate / type → Z3 formula translation
# ---------------------------------------------------------------------------


class SMTEncoder:
    """Encodes the predicate language and refinement types into Z3 formulas.

    Every predicate ``φ`` is translated into a ``z3.BoolRef`` and every
    arithmetic expression into a ``z3.ArithRef``.
    """

    def __init__(self, context: SMTContext) -> None:
        self.ctx = context

    # -- public entry point --------------------------------------------------

    def encode_predicate(self, pred: Predicate) -> z3.BoolRef:
        """Dispatch on predicate type and encode to Z3."""
        if isinstance(pred, TruePred):
            return z3.BoolVal(True, ctx=self.ctx.ctx)
        if isinstance(pred, FalsePred):
            return z3.BoolVal(False, ctx=self.ctx.ctx)
        if isinstance(pred, ComparisonPred):
            return self.encode_comparison(pred)
        if isinstance(pred, TypeTagPred):
            return self.encode_type_tag(pred)
        if isinstance(pred, NullityPred):
            return self.encode_nullity(pred)
        if isinstance(pred, TruthinessPred):
            return self.encode_truthiness(pred)
        if isinstance(pred, HasAttrPred):
            return self.encode_hasattr(pred)
        if isinstance(pred, AndPred):
            return self.encode_conjunction(pred)
        if isinstance(pred, OrPred):
            return self.encode_disjunction(pred)
        if isinstance(pred, NotPred):
            return self.encode_negation(pred)
        raise TypeError(f"Unsupported predicate: {type(pred).__name__}")

    # -- comparison ----------------------------------------------------------

    def encode_comparison(self, pred: ComparisonPred) -> z3.BoolRef:
        """Encode e₁ ⊕ e₂ as a Z3 boolean expression."""
        left = self.encode_linear_expr(pred.left)
        right = self.encode_linear_expr(pred.right)
        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }
        if pred.op not in ops:
            raise ValueError(f"Unknown comparison operator: {pred.op!r}")
        return ops[pred.op](left, right)

    # -- type tag ------------------------------------------------------------

    def encode_type_tag(self, pred: TypeTagPred) -> z3.BoolRef:
        """Encode tag(x) == T as a Z3 equality over the Tag sort."""
        var = self.ctx.get_var(pred.var)
        tag_val = self.ctx.tag_fn(var)
        tag_const = self.ctx.tag_const(pred.tag)
        return tag_val == tag_const

    # -- nullity -------------------------------------------------------------

    def encode_nullity(self, pred: NullityPred) -> z3.BoolRef:
        """Encode is_none(x) or ¬is_none(x)."""
        var = self.ctx.get_var(pred.var)
        is_none = self.ctx.is_none_fn(var)
        if pred.is_null:
            return is_none == z3.BoolVal(True, ctx=self.ctx.ctx)
        return is_none == z3.BoolVal(False, ctx=self.ctx.ctx)

    # -- truthiness ----------------------------------------------------------

    def encode_truthiness(self, pred: TruthinessPred) -> z3.BoolRef:
        """Encode is_truthy(x) or ¬is_truthy(x)."""
        var = self.ctx.get_var(pred.var)
        truthy = self.ctx.is_truthy_fn(var)
        if pred.is_truthy:
            return truthy == z3.BoolVal(True, ctx=self.ctx.ctx)
        return truthy == z3.BoolVal(False, ctx=self.ctx.ctx)

    # -- hasattr -------------------------------------------------------------

    def encode_hasattr(self, pred: HasAttrPred) -> z3.BoolRef:
        """Encode hasattr(x, "k") using the interned attribute integer."""
        var = self.ctx.get_var(pred.var)
        attr_int = self.ctx.intern_attr(pred.attr)
        has = self.ctx.hasattr_fn(var, attr_int)
        return has == z3.BoolVal(True, ctx=self.ctx.ctx)

    # -- compound predicates -------------------------------------------------

    def encode_conjunction(self, pred: AndPred) -> z3.BoolRef:
        """Encode φ₁ ∧ φ₂ ∧ … as a Z3 And."""
        if not pred.conjuncts:
            return z3.BoolVal(True, ctx=self.ctx.ctx)
        encoded = [self.encode_predicate(c) for c in pred.conjuncts]
        if len(encoded) == 1:
            return encoded[0]
        return z3.And(*encoded)

    def encode_disjunction(self, pred: OrPred) -> z3.BoolRef:
        """Encode φ₁ ∨ φ₂ ∨ … as a Z3 Or."""
        if not pred.disjuncts:
            return z3.BoolVal(False, ctx=self.ctx.ctx)
        encoded = [self.encode_predicate(d) for d in pred.disjuncts]
        if len(encoded) == 1:
            return encoded[0]
        return z3.Or(*encoded)

    def encode_negation(self, pred: NotPred) -> z3.BoolRef:
        """Encode ¬φ as a Z3 Not."""
        return z3.Not(self.encode_predicate(pred.inner))

    # -- arithmetic expressions ----------------------------------------------

    def encode_linear_expr(self, expr: LinearExpr) -> z3.ArithRef:
        """Encode an arithmetic expression into Z3."""
        if isinstance(expr, VarExpr):
            return self.ctx.get_var(expr.name)
        if isinstance(expr, ConstExpr):
            return z3.IntVal(expr.value, ctx=self.ctx.ctx)
        if isinstance(expr, LenExpr):
            arg = self.ctx.get_var(expr.arg)
            return self.ctx.len_fn(arg)
        if isinstance(expr, BinOpExpr):
            left = self.encode_linear_expr(expr.left)
            right = self.encode_linear_expr(expr.right)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            raise ValueError(f"Unknown binary operator: {expr.op!r}")
        if isinstance(expr, ScaleExpr):
            inner = self.encode_linear_expr(expr.expr)
            c = z3.IntVal(expr.const, ctx=self.ctx.ctx)
            if expr.op == "*":
                return inner * c
            if expr.op == "//":
                return inner / c  # Z3 integer division
            if expr.op == "%":
                return inner % c
            raise ValueError(f"Unknown scale operator: {expr.op!r}")
        raise TypeError(f"Unsupported expression: {type(expr).__name__}")

    # -- refinement type encoding --------------------------------------------

    def encode_refinement_type(self, rtype: RefinementType) -> z3.BoolRef:
        """Encode {x : τ | φ} as the Z3 formula for φ[x ↦ z3_var(binder)].

        The base type is encoded as a tag constraint when applicable.
        """
        pred_formula = self.encode_predicate(rtype.pred)
        base_constraint = self._encode_base_type_constraint(
            rtype.binder, rtype.base
        )
        if base_constraint is not None:
            return z3.And(base_constraint, pred_formula)
        return pred_formula

    def _encode_base_type_constraint(
        self, binder: str, base: BaseType
    ) -> Optional[z3.BoolRef]:
        """Optionally encode a base type as a tag constraint."""
        tag_name = self._base_type_to_tag(base)
        if tag_name is not None and self.ctx.has_tag(tag_name):
            var = self.ctx.get_var(binder)
            return self.ctx.tag_fn(var) == self.ctx.tag_const(tag_name)
        return None

    @staticmethod
    def _base_type_to_tag(base: BaseType) -> Optional[str]:
        """Map a BaseType to its tag name, if applicable."""
        if isinstance(base, IntType):
            return "int"
        if isinstance(base, StrType):
            return "str"
        if isinstance(base, BoolType):
            return "bool"
        if isinstance(base, NoneType_):
            return "NoneType"
        if isinstance(base, ListType):
            return "list"
        return None


# ---------------------------------------------------------------------------
# SubtypingQuery – Γ ⊢ {x:τ₁|φ₁} <: {x:τ₂|φ₂}
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtypingQuery:
    """A refinement subtyping judgment.

    Encoded as: ∀x. (Γ ∧ φ₁) → φ₂
    For satisfiability checking we negate:  ∃x. Γ ∧ φ₁ ∧ ¬φ₂
    If UNSAT the subtyping holds; if SAT the model is a counterexample.
    """

    sub: RefinementType
    sup: RefinementType
    env_constraints: Sequence[Predicate] = ()

    def encode(self, encoder: SMTEncoder) -> z3.BoolRef:
        """Encode the *negated* query for a satisfiability check.

        Returns ∃x. Γ ∧ φ₁ ∧ ¬φ₂  (existential via Skolemization).
        """
        binder = self.sub.binder

        # Ensure the supertype uses the same binder
        sup_renamed = self.sup
        if sup_renamed.binder != binder:
            sup_renamed = sup_renamed.alpha_rename(binder)

        phi1 = encoder.encode_predicate(self.sub.pred)
        phi2 = encoder.encode_predicate(sup_renamed.pred)

        # Γ: environment constraints
        gamma_parts: List[z3.BoolRef] = []
        for ec in self.env_constraints:
            gamma_parts.append(encoder.encode_predicate(ec))

        # Base-type constraints (tag equality)
        base1 = encoder._encode_base_type_constraint(binder, self.sub.base)
        if base1 is not None:
            gamma_parts.append(base1)

        gamma = z3.And(*gamma_parts) if gamma_parts else z3.BoolVal(True)

        # Negated query: Γ ∧ φ₁ ∧ ¬φ₂
        return z3.And(gamma, phi1, z3.Not(phi2))

    def encode_validity(self, encoder: SMTEncoder) -> z3.BoolRef:
        """Encode as the *positive* implication (Γ ∧ φ₁) → φ₂."""
        binder = self.sub.binder
        sup_renamed = self.sup
        if sup_renamed.binder != binder:
            sup_renamed = sup_renamed.alpha_rename(binder)

        phi1 = encoder.encode_predicate(self.sub.pred)
        phi2 = encoder.encode_predicate(sup_renamed.pred)

        gamma_parts: List[z3.BoolRef] = []
        for ec in self.env_constraints:
            gamma_parts.append(encoder.encode_predicate(ec))

        base1 = encoder._encode_base_type_constraint(binder, self.sub.base)
        if base1 is not None:
            gamma_parts.append(base1)

        gamma = z3.And(*gamma_parts) if gamma_parts else z3.BoolVal(True)
        return z3.Implies(z3.And(gamma, phi1), phi2)


# ---------------------------------------------------------------------------
# SMTSolver – wraps Z3 with management, caching, incremental solving
# ---------------------------------------------------------------------------


class SMTSolver:
    """High-level solver interface over Z3.

    Provides satisfiability, validity, entailment checks, model extraction,
    unsat-core retrieval, and incremental push/pop solving.
    """

    def __init__(
        self,
        context: Optional[SMTContext] = None,
        *,
        default_timeout_ms: int = 5000,
    ) -> None:
        self.context = context or SMTContext()
        self.encoder = SMTEncoder(self.context)
        self._solver = z3.Solver(ctx=self.context.ctx)
        self._default_timeout = default_timeout_ms
        self._solver.set("timeout", default_timeout_ms)

    # -- satisfiability ------------------------------------------------------

    def check_sat(
        self,
        formula: z3.BoolRef,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        """Check satisfiability of *formula*."""
        if timeout_ms is not None:
            self._solver.set("timeout", timeout_ms)
        else:
            self._solver.set("timeout", self._default_timeout)

        self._solver.push()
        self._solver.add(formula)
        t0 = time.monotonic()
        result = self._solver.check()
        elapsed = time.monotonic() - t0

        status = self._z3_result_to_status(result)
        model_dict: Optional[Dict[str, Any]] = None
        if status is Satisfiability.SAT:
            model_dict = self._extract_model(self._solver.model())

        self._solver.pop()
        return SMTResult(
            status=status,
            model=model_dict,
            time_taken=elapsed,
        )

    # -- validity ------------------------------------------------------------

    def check_validity(
        self,
        formula: z3.BoolRef,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        """Check validity: *formula* is valid iff ¬formula is UNSAT."""
        neg = z3.Not(formula)
        result = self.check_sat(neg, timeout_ms)
        # Flip the semantics: UNSAT of negation means VALID
        if result.is_unsat:
            return SMTResult(
                status=Satisfiability.UNSAT,  # "valid" represented as UNSAT-of-negation
                time_taken=result.time_taken,
            )
        if result.is_sat:
            return SMTResult(
                status=Satisfiability.SAT,
                model=result.model,
                time_taken=result.time_taken,
            )
        return SMTResult(
            status=Satisfiability.UNKNOWN,
            time_taken=result.time_taken,
        )

    # -- entailment ----------------------------------------------------------

    def check_entailment(
        self,
        premises: Sequence[z3.BoolRef],
        conclusion: z3.BoolRef,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        """Check whether *premises* ⊨ *conclusion*.

        Encoded as: premises ∧ ¬conclusion  is UNSAT?
        """
        conj = z3.And(*premises) if premises else z3.BoolVal(True)
        formula = z3.And(conj, z3.Not(conclusion))
        return self.check_sat(formula, timeout_ms)

    # -- model extraction ----------------------------------------------------

    def get_model(self) -> Dict[str, Any]:
        """Return the model from the last satisfiable check.

        Must be called immediately after a SAT check_sat (before pop).
        Falls back to an empty dict if no model is available.
        """
        try:
            m = self._solver.model()
            return self._extract_model(m)
        except z3.Z3Exception:
            return {}

    def _extract_model(self, model: z3.ModelRef) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for decl in model.decls():
            name = decl.name()
            val = model[decl]
            if z3.is_int_value(val):
                result[name] = val.as_long()
            elif z3.is_true(val):
                result[name] = True
            elif z3.is_false(val):
                result[name] = False
            elif z3.is_rational_value(val):
                result[name] = float(val.as_fraction())
            else:
                result[name] = str(val)
        return result

    # -- unsat-core-based predicate extraction ---------------------------------

    def get_interpolant(
        self,
        formula_a: z3.BoolRef,
        formula_b: z3.BoolRef,
    ) -> Optional[z3.BoolRef]:
        """Attempt unsat-core-based predicate extraction for A ∧ B ⊨ ⊥.

        Tries Z3's built-in interpolation when available, otherwise falls
        back to unsat-core-based approximation.  Note: the result does NOT
        satisfy the vocabulary restriction of Craig interpolants.
        """
        try:
            interp_result = z3.Interpolant(formula_a)
            s = z3.Solver(ctx=self.context.ctx)
            s.set("proof", True)
            s.add(interp_result)
            s.add(formula_b)
            if s.check() == z3.unsat:
                proof = s.proof()
                interps = z3.get_interpolant(proof, interp_result, [])
                if interps:
                    return interps[0]
        except (z3.Z3Exception, AttributeError):
            pass

        # Fallback: use unsat core as an over-approximation
        s = z3.Solver(ctx=self.context.ctx)
        s.set("unsat_core", True)
        a_label = z3.Bool("__interp_a__")
        b_label = z3.Bool("__interp_b__")
        s.assert_and_track(formula_a, a_label)
        s.assert_and_track(formula_b, b_label)
        if s.check() == z3.unsat:
            core = s.unsat_core()
            if a_label in core:
                return formula_a
            return z3.BoolVal(False)
        return None

    # -- unsat core ----------------------------------------------------------

    def get_unsat_core(self) -> List[z3.ExprRef]:
        """Return the unsat core from the last UNSAT check.

        Requires that assertions were added via assert_and_track.
        """
        try:
            return list(self._solver.unsat_core())
        except z3.Z3Exception:
            return []

    def check_with_core(
        self,
        labeled_assertions: Dict[str, z3.BoolRef],
        extra: Optional[z3.BoolRef] = None,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[SMTResult, List[str]]:
        """Check with labeled assertions and return unsat core labels."""
        if timeout_ms is not None:
            self._solver.set("timeout", timeout_ms)
        self._solver.set("unsat_core", True)

        self._solver.push()
        label_map: Dict[str, z3.BoolRef] = {}
        for name, assertion in labeled_assertions.items():
            label = z3.Bool(f"__core_{name}__")
            label_map[name] = label
            self._solver.assert_and_track(assertion, label)
        if extra is not None:
            self._solver.add(extra)

        t0 = time.monotonic()
        result = self._solver.check()
        elapsed = time.monotonic() - t0

        status = self._z3_result_to_status(result)
        core_labels: List[str] = []
        if status is Satisfiability.UNSAT:
            core = self._solver.unsat_core()
            for name, label in label_map.items():
                if label in core:
                    core_labels.append(name)

        self._solver.pop()
        return (
            SMTResult(status=status, time_taken=elapsed),
            core_labels,
        )

    # -- incremental solving -------------------------------------------------

    def incremental_check(
        self,
        base_assertions: Sequence[z3.BoolRef],
        queries: Sequence[z3.BoolRef],
        timeout_ms: Optional[int] = None,
    ) -> List[SMTResult]:
        """Incrementally check multiple queries sharing a common base.

        Uses push/pop to avoid re-encoding the base for each query.
        """
        if timeout_ms is not None:
            self._solver.set("timeout", timeout_ms)

        self._solver.push()
        for a in base_assertions:
            self._solver.add(a)

        results: List[SMTResult] = []
        for q in queries:
            self._solver.push()
            self._solver.add(q)
            t0 = time.monotonic()
            r = self._solver.check()
            elapsed = time.monotonic() - t0
            status = self._z3_result_to_status(r)
            model_dict = None
            if status is Satisfiability.SAT:
                model_dict = self._extract_model(self._solver.model())
            results.append(
                SMTResult(status=status, model=model_dict, time_taken=elapsed)
            )
            self._solver.pop()

        self._solver.pop()
        return results

    # -- subtyping -----------------------------------------------------------

    def check_subtyping(
        self,
        query: SubtypingQuery,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        """Check refinement subtyping: UNSAT ⇒ subtype holds."""
        negated = query.encode(self.encoder)
        return self.check_sat(negated, timeout_ms)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _z3_result_to_status(result: z3.CheckSatResult) -> Satisfiability:
        if result == z3.sat:
            return Satisfiability.SAT
        if result == z3.unsat:
            return Satisfiability.UNSAT
        return Satisfiability.UNKNOWN

    def reset(self) -> None:
        """Reset the solver and context."""
        self._solver.reset()
        self.context.reset()

    def __repr__(self) -> str:
        return f"SMTSolver(timeout={self._default_timeout}ms)"


# ---------------------------------------------------------------------------
# CounterexampleEncoder – encode counterexample traces as SMT formulas
# ---------------------------------------------------------------------------


class CounterexampleEncoder:
    """Encodes execution traces and safety properties for feasibility checks."""

    def __init__(self, encoder: SMTEncoder) -> None:
        self.encoder = encoder
        self.ctx = encoder.ctx
        self._step: int = 0

    def _step_var(self, name: str) -> z3.ArithRef:
        """Create a step-indexed variable: name@step."""
        return self.ctx.get_var(f"{name}@{self._step}")

    def encode_path(self, path: Sequence[Any]) -> z3.BoolRef:
        """Encode a counterexample path as a conjunction of step constraints.

        Each node in *path* should expose ``get_predicates()`` returning
        a list of ``Predicate`` objects, or be a ``Predicate`` directly.
        """
        constraints: List[z3.BoolRef] = []
        for node in path:
            if isinstance(node, Predicate):
                constraints.append(self.encoder.encode_predicate(node))
            elif hasattr(node, "get_predicates"):
                for pred in node.get_predicates():
                    constraints.append(self.encoder.encode_predicate(pred))
            elif hasattr(node, "condition") and isinstance(
                node.condition, Predicate
            ):
                constraints.append(
                    self.encoder.encode_predicate(node.condition)
                )
            self._step += 1
        if not constraints:
            return z3.BoolVal(True, ctx=self.ctx.ctx)
        return z3.And(*constraints)

    def encode_state(self, state: Any) -> z3.BoolRef:
        """Encode an abstract state as a conjunction of variable constraints.

        *state* is expected to have an ``items()`` method yielding
        ``(var_name, abstract_value)`` pairs.
        """
        constraints: List[z3.BoolRef] = []
        if hasattr(state, "items"):
            for var_name, abstract_val in state.items():
                z3_var = self.ctx.get_var(var_name)
                if isinstance(abstract_val, int):
                    constraints.append(z3_var == z3.IntVal(abstract_val))
                elif isinstance(abstract_val, bool):
                    constraints.append(
                        self.ctx.is_truthy_fn(z3_var)
                        == z3.BoolVal(abstract_val)
                    )
                elif isinstance(abstract_val, Predicate):
                    constraints.append(
                        self.encoder.encode_predicate(abstract_val)
                    )
        if hasattr(state, "is_bottom") and state.is_bottom:
            return z3.BoolVal(False, ctx=self.ctx.ctx)
        if not constraints:
            return z3.BoolVal(True, ctx=self.ctx.ctx)
        return z3.And(*constraints)

    def encode_safety_property(self, prop: Any) -> z3.BoolRef:
        """Encode a safety property.

        *prop* can be a ``Predicate``, a ``z3.BoolRef``, or an object
        with a ``predicate`` attribute.
        """
        if isinstance(prop, Predicate):
            return self.encoder.encode_predicate(prop)
        if isinstance(prop, z3.BoolRef):
            return prop
        if hasattr(prop, "predicate") and isinstance(
            prop.predicate, Predicate
        ):
            return self.encoder.encode_predicate(prop.predicate)
        raise TypeError(
            f"Cannot encode safety property of type {type(prop).__name__}"
        )

    def check_feasibility(
        self,
        path_condition: z3.BoolRef,
        safety_negation: z3.BoolRef,
        solver: Optional[SMTSolver] = None,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        """Check whether a path violating safety is feasible.

        Returns SAT if a real counterexample exists.
        """
        s = solver or SMTSolver(self.ctx)
        formula = z3.And(path_condition, safety_negation)
        return s.check_sat(formula, timeout_ms)

    def reset_steps(self) -> None:
        self._step = 0


# ---------------------------------------------------------------------------
# InterpolantExtractor – unsat-core-based predicate extraction
# ---------------------------------------------------------------------------


class InterpolantExtractor:
    """Extracts predicates from UNSAT pairs via unsat-core extraction
    and projects them back to the predicate template language.
    Note: despite the name, this uses unsat cores, not true Craig interpolants.
    """

    def __init__(self, encoder: SMTEncoder) -> None:
        self.encoder = encoder
        self.ctx = encoder.ctx

    def extract(
        self,
        formula_a: z3.BoolRef,
        formula_b: z3.BoolRef,
    ) -> Optional[z3.BoolRef]:
        """Extract a predicate from A ∧ B ⊨ ⊥ via unsat core."""
        solver = SMTSolver(self.ctx)
        return solver.get_interpolant(formula_a, formula_b)

    def project_to_predicates(
        self,
        interpolant: z3.BoolRef,
    ) -> List[Predicate]:
        """Project a Z3 interpolant back to the predicate template language.

        Performs a best-effort structural decomposition.
        """
        simplified = z3.simplify(interpolant)
        return self._z3_to_predicates(simplified)

    def _z3_to_predicates(self, expr: z3.ExprRef) -> List[Predicate]:
        """Recursively convert a Z3 expression to predicates."""
        predicates: List[Predicate] = []

        if z3.is_and(expr):
            for child in expr.children():
                predicates.extend(self._z3_to_predicates(child))
            return predicates

        if z3.is_or(expr):
            sub_preds: List[Predicate] = []
            for child in expr.children():
                child_preds = self._z3_to_predicates(child)
                if len(child_preds) == 1:
                    sub_preds.append(child_preds[0])
                elif child_preds:
                    sub_preds.append(
                        AndPred(frozenset(child_preds))
                    )
            if sub_preds:
                predicates.append(OrPred(frozenset(sub_preds)))
            return predicates

        if z3.is_not(expr):
            inner = self._z3_to_predicates(expr.children()[0])
            for p in inner:
                predicates.append(NotPred(p))
            return predicates

        # Attempt to recognize comparison patterns
        comp = self._try_comparison(expr)
        if comp is not None:
            predicates.append(comp)
            return predicates

        # Attempt tag equality
        tag = self._try_tag_pred(expr)
        if tag is not None:
            predicates.append(tag)
            return predicates

        # Fallback: wrap the Z3 expression as a TruePred (unknown structure)
        predicates.append(TruePred())
        return predicates

    def _try_comparison(self, expr: z3.ExprRef) -> Optional[ComparisonPred]:
        """Try to recognize a comparison expression."""
        decl = expr.decl()
        kind = decl.kind()
        op_map = {
            z3.Z3_OP_EQ: "==",
            z3.Z3_OP_LE: "<=",
            z3.Z3_OP_GE: ">=",
            z3.Z3_OP_LT: "<",
            z3.Z3_OP_GT: ">",
        }
        if kind not in op_map:
            # Also handle distinct (!=)
            if kind == z3.Z3_OP_DISTINCT:
                children = expr.children()
                if len(children) == 2:
                    left = self._z3_to_linear(children[0])
                    right = self._z3_to_linear(children[1])
                    if left is not None and right is not None:
                        return ComparisonPred("!=", left, right)
            return None
        children = expr.children()
        if len(children) != 2:
            return None
        left = self._z3_to_linear(children[0])
        right = self._z3_to_linear(children[1])
        if left is None or right is None:
            return None
        return ComparisonPred(op_map[kind], left, right)

    def _try_tag_pred(self, expr: z3.ExprRef) -> Optional[TypeTagPred]:
        """Try to recognize tag(x) == T."""
        if not z3.is_eq(expr):
            return None
        children = expr.children()
        if len(children) != 2:
            return None
        lhs, rhs = children
        if (
            z3.is_app(lhs)
            and lhs.decl().name() == "tag"
            and lhs.num_args() == 1
        ):
            arg = lhs.arg(0)
            if z3.is_const(arg) and arg.sort() == self.ctx.int_sort:
                var_name = str(arg)
                tag_name = str(rhs)
                return TypeTagPred(var_name, tag_name)
        return None

    def _z3_to_linear(self, expr: z3.ExprRef) -> Optional[LinearExpr]:
        """Convert a Z3 arithmetic expression back to LinearExpr."""
        if z3.is_int_value(expr):
            return ConstExpr(expr.as_long())
        if z3.is_const(expr) and expr.sort() == self.ctx.int_sort:
            return VarExpr(str(expr))
        if z3.is_app(expr):
            decl = expr.decl()
            kind = decl.kind()
            children = expr.children()
            if kind == z3.Z3_OP_ADD and len(children) == 2:
                left = self._z3_to_linear(children[0])
                right = self._z3_to_linear(children[1])
                if left is not None and right is not None:
                    return BinOpExpr("+", left, right)
            if kind == z3.Z3_OP_SUB and len(children) == 2:
                left = self._z3_to_linear(children[0])
                right = self._z3_to_linear(children[1])
                if left is not None and right is not None:
                    return BinOpExpr("-", left, right)
            if kind == z3.Z3_OP_MUL and len(children) == 2:
                left = self._z3_to_linear(children[0])
                right = self._z3_to_linear(children[1])
                if left is not None and right is not None:
                    if isinstance(right, ConstExpr):
                        return ScaleExpr("*", left, right.value)
                    if isinstance(left, ConstExpr):
                        return ScaleExpr("*", right, left.value)
            if decl.name() == "len" and len(children) == 1:
                arg = children[0]
                if z3.is_const(arg):
                    return LenExpr(str(arg))
        return None

    def simplify_interpolant(self, interp: z3.BoolRef) -> z3.BoolRef:
        """Apply Z3 simplification tactics to an interpolant."""
        goal = z3.Goal()
        goal.add(interp)
        tactic = z3.Then(
            z3.Tactic("simplify"),
            z3.Tactic("propagate-values"),
            z3.Tactic("ctx-simplify"),
        )
        result = tactic(goal)
        if len(result) == 1 and len(result[0]) > 0:
            formulas = [result[0][i] for i in range(len(result[0]))]
            if len(formulas) == 1:
                return formulas[0]
            return z3.And(*formulas)
        return interp


# ---------------------------------------------------------------------------
# LemmaCacher – caches SMT results for similar queries
# ---------------------------------------------------------------------------


class LemmaCacher:
    """Caches SMT query results keyed by a structural hash of the formula.

    Avoids re-solving identical or α-equivalent queries.
    """

    def __init__(self, max_size: int = 8192) -> None:
        self._cache: Dict[str, SMTResult] = {}
        self._max_size = max_size
        self._hits: int = 0
        self._misses: int = 0

    def cache_key(self, query: Union[z3.BoolRef, SubtypingQuery]) -> str:
        """Produce a hashable cache key from a query."""
        if isinstance(query, SubtypingQuery):
            raw = (
                f"subtype:{query.sub.pretty()}:"
                f"{query.sup.pretty()}:"
                f"{[e.pretty() for e in query.env_constraints]}"
            )
        else:
            raw = str(query.sexpr()) if hasattr(query, "sexpr") else str(query)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def lookup(self, key: str) -> Optional[SMTResult]:
        """Look up a cached result."""
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def store(self, key: str, result: SMTResult) -> None:
        """Store a result in the cache, evicting oldest if full."""
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = result

    def invalidate(self, affected_keys: Sequence[str]) -> None:
        """Remove specific keys from the cache."""
        for key in affected_keys:
            self._cache.pop(key, None)

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
            f"LemmaCacher(size={self.size}, "
            f"hits={self._hits}, misses={self._misses}, "
            f"rate={self.hit_rate:.2%})"
        )


# ---------------------------------------------------------------------------
# ModelMinimizer – minimize SAT models for readable counterexamples
# ---------------------------------------------------------------------------


class ModelMinimizer:
    """Minimizes a satisfying model by attempting to set variables to
    small canonical values while preserving satisfiability.
    """

    # Candidate values tried during minimization, in preference order
    CANDIDATES: Sequence[int] = (0, 1, -1, 2, -2, 3, 10, 100)

    def __init__(self, solver: SMTSolver) -> None:
        self.solver = solver

    def minimize(
        self,
        model: Dict[str, Any],
        formula: z3.BoolRef,
    ) -> Dict[str, Any]:
        """Attempt to minimize each integer assignment in *model*.

        For each integer variable tries small canonical values and keeps
        the smallest one that preserves satisfiability.
        """
        minimized = dict(model)
        int_vars = [
            (k, v) for k, v in model.items() if isinstance(v, int)
        ]
        # Sort by absolute value descending – minimize large values first
        int_vars.sort(key=lambda kv: abs(kv[1]), reverse=True)

        for var_name, current_val in int_vars:
            z3_var = self.solver.context.get_var(var_name)
            for candidate in self.CANDIDATES:
                if candidate == current_val:
                    break
                # Build the formula with this variable pinned
                pinned = z3.And(formula, z3_var == z3.IntVal(candidate))
                result = self.solver.check_sat(pinned, timeout_ms=500)
                if result.is_sat:
                    minimized[var_name] = candidate
                    break
        return minimized

    def extract_relevant_assignments(
        self,
        model: Dict[str, Any],
        variables_of_interest: Set[str],
    ) -> Dict[str, Any]:
        """Filter model to only the variables of interest."""
        return {
            k: v for k, v in model.items() if k in variables_of_interest
        }


# ---------------------------------------------------------------------------
# FormulaSimplifier – simplify Z3 formulas before solving
# ---------------------------------------------------------------------------


class FormulaSimplifier:
    """Pre-processing simplifications for Z3 formulas."""

    def simplify(self, formula: z3.BoolRef) -> z3.BoolRef:
        """Apply Z3's built-in simplifier."""
        return z3.simplify(formula)

    def constant_fold(self, formula: z3.BoolRef) -> z3.BoolRef:
        """Constant propagation and folding via Z3 tactics."""
        goal = z3.Goal()
        goal.add(formula)
        tactic = z3.Then(
            z3.Tactic("simplify"),
            z3.Tactic("propagate-values"),
        )
        result = tactic(goal)
        return self._goal_result_to_formula(result)

    def remove_trivial(self, formula: z3.BoolRef) -> z3.BoolRef:
        """Remove trivially true/false sub-formulas."""
        goal = z3.Goal()
        goal.add(formula)
        tactic = z3.Then(
            z3.Tactic("simplify"),
            z3.Tactic("ctx-simplify"),
            z3.Tactic("elim-uncnstr"),
        )
        result = tactic(goal)
        return self._goal_result_to_formula(result)

    def full_simplify(self, formula: z3.BoolRef) -> z3.BoolRef:
        """Aggressive simplification pipeline."""
        goal = z3.Goal()
        goal.add(formula)
        tactic = z3.Then(
            z3.Tactic("simplify"),
            z3.Tactic("propagate-values"),
            z3.Tactic("ctx-simplify"),
            z3.Tactic("elim-uncnstr"),
            z3.Tactic("simplify"),
        )
        result = tactic(goal)
        return self._goal_result_to_formula(result)

    @staticmethod
    def _goal_result_to_formula(result: z3.ApplyResult) -> z3.BoolRef:
        """Convert a tactic ApplyResult back to a single formula."""
        formulas: List[z3.BoolRef] = []
        for subgoal in result:
            for i in range(len(subgoal)):
                formulas.append(subgoal[i])
        if not formulas:
            return z3.BoolVal(True)
        if len(formulas) == 1:
            return formulas[0]
        return z3.And(*formulas)


# ---------------------------------------------------------------------------
# SMTLIBExporter – export formulas as SMT-LIB 2.0 strings
# ---------------------------------------------------------------------------


class SMTLIBExporter:
    """Export Z3 formulas as SMT-LIB 2.0 format for proof certificates
    and external solver compatibility.
    """

    def __init__(self, context: Optional[SMTContext] = None) -> None:
        self.context = context

    def export(self, formula: z3.BoolRef) -> str:
        """Export *formula* as an SMT-LIB string (declarations + assert)."""
        s = z3.Solver()
        s.add(formula)
        return s.sexpr()

    def export_with_check(self, formula: z3.BoolRef) -> str:
        """Export with (check-sat) and (get-model) commands."""
        lines: List[str] = []
        lines.append("; SMT-LIB 2.0 export")
        lines.append("(set-logic QF_UFLIA)")
        lines.append("")

        # Collect free constants and their sorts
        consts = self._collect_constants(formula)
        for name, sort in sorted(consts.items()):
            lines.append(f"(declare-const {name} {sort})")
        lines.append("")

        # Collect uninterpreted function declarations
        funcs = self._collect_functions(formula)
        for name, (arg_sorts, ret_sort) in sorted(funcs.items()):
            args = " ".join(arg_sorts)
            lines.append(f"(declare-fun {name} ({args}) {ret_sort})")
        lines.append("")

        lines.append(f"(assert")
        lines.append(f"  {formula.sexpr()}")
        lines.append(f")")
        lines.append("")
        lines.append("(check-sat)")
        lines.append("(get-model)")
        return "\n".join(lines)

    def export_subtyping_query(
        self,
        query: SubtypingQuery,
        encoder: SMTEncoder,
    ) -> str:
        """Export a subtyping query as an SMT-LIB string."""
        negated = query.encode(encoder)
        lines: List[str] = []
        lines.append("; Refinement subtyping query")
        lines.append(f"; {query.sub.pretty()} <: {query.sup.pretty()}")
        lines.append("; UNSAT ⇒ subtyping holds")
        lines.append("")
        lines.append(self.export_with_check(negated))
        return "\n".join(lines)

    def _collect_constants(
        self, formula: z3.ExprRef
    ) -> Dict[str, str]:
        """Collect all free constants in the formula."""
        consts: Dict[str, str] = {}
        self._walk_consts(formula, consts, set())
        return consts

    def _walk_consts(
        self,
        expr: z3.ExprRef,
        consts: Dict[str, str],
        visited: Set[int],
    ) -> None:
        expr_id = expr.get_id()
        if expr_id in visited:
            return
        visited.add(expr_id)

        if z3.is_const(expr) and expr.decl().arity() == 0:
            name = expr.decl().name()
            if not name.startswith("__") and not z3.is_int_value(expr):
                sort = expr.sort()
                if sort == z3.IntSort():
                    consts[name] = "Int"
                elif sort == z3.BoolSort():
                    consts[name] = "Bool"
                else:
                    consts[name] = str(sort)
        for child in expr.children():
            self._walk_consts(child, consts, visited)

    def _collect_functions(
        self, formula: z3.ExprRef
    ) -> Dict[str, Tuple[List[str], str]]:
        """Collect uninterpreted function declarations."""
        funcs: Dict[str, Tuple[List[str], str]] = {}
        self._walk_funcs(formula, funcs, set())
        return funcs

    def _walk_funcs(
        self,
        expr: z3.ExprRef,
        funcs: Dict[str, Tuple[List[str], str]],
        visited: Set[int],
    ) -> None:
        expr_id = expr.get_id()
        if expr_id in visited:
            return
        visited.add(expr_id)

        if z3.is_app(expr):
            decl = expr.decl()
            arity = decl.arity()
            if arity > 0 and decl.kind() == z3.Z3_OP_UNINTERPRETED:
                name = decl.name()
                arg_sorts = [str(decl.domain(i)) for i in range(arity)]
                ret_sort = str(decl.range())
                funcs[name] = (arg_sorts, ret_sort)
        for child in expr.children():
            self._walk_funcs(child, funcs, visited)


# ---------------------------------------------------------------------------
# CachingSMTSolver – solver with integrated lemma caching
# ---------------------------------------------------------------------------


class CachingSMTSolver:
    """SMTSolver with integrated LemmaCacher for automatic result caching."""

    def __init__(
        self,
        context: Optional[SMTContext] = None,
        *,
        default_timeout_ms: int = 5000,
        cache_size: int = 8192,
    ) -> None:
        self.solver = SMTSolver(
            context, default_timeout_ms=default_timeout_ms
        )
        self.cache = LemmaCacher(max_size=cache_size)

    @property
    def context(self) -> SMTContext:
        return self.solver.context

    @property
    def encoder(self) -> SMTEncoder:
        return self.solver.encoder

    def check_sat(
        self,
        formula: z3.BoolRef,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        key = self.cache.cache_key(formula)
        cached = self.cache.lookup(key)
        if cached is not None:
            return cached
        result = self.solver.check_sat(formula, timeout_ms)
        self.cache.store(key, result)
        return result

    def check_subtyping(
        self,
        query: SubtypingQuery,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        key = self.cache.cache_key(query)
        cached = self.cache.lookup(key)
        if cached is not None:
            return cached
        result = self.solver.check_subtyping(query, timeout_ms)
        self.cache.store(key, result)
        return result

    def check_entailment(
        self,
        premises: Sequence[z3.BoolRef],
        conclusion: z3.BoolRef,
        timeout_ms: Optional[int] = None,
    ) -> SMTResult:
        return self.solver.check_entailment(premises, conclusion, timeout_ms)

    def __repr__(self) -> str:
        return f"CachingSMTSolver({self.solver!r}, cache={self.cache!r})"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_solver(
    *,
    tags: Sequence[str] = _DEFAULT_TAGS,
    timeout_ms: int = 5000,
    caching: bool = True,
    cache_size: int = 8192,
) -> Union[CachingSMTSolver, SMTSolver]:
    """Create a configured solver instance."""
    ctx = SMTContext(tag_names=tags)
    if caching:
        return CachingSMTSolver(
            ctx, default_timeout_ms=timeout_ms, cache_size=cache_size
        )
    return SMTSolver(ctx, default_timeout_ms=timeout_ms)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "Satisfiability",
    "SMTResult",
    "SMTContext",
    "SMTEncoder",
    "SubtypingQuery",
    "SMTSolver",
    "CounterexampleEncoder",
    "InterpolantExtractor",
    "LemmaCacher",
    "ModelMinimizer",
    "FormulaSimplifier",
    "SMTLIBExporter",
    "CachingSMTSolver",
    "make_solver",
]
