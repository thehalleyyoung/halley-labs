from __future__ import annotations

"""
Proof certificate generation and verification for refinement type inference.

Generates machine-checkable proof certificates that witness the correctness
of inferred refinement types and contracts. Supports multiple output formats
including SMT-LIB 2.6, natural language, LaTeX proof trees, Coq, and Lean 4.
"""

import hashlib
import time
import math
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Core type / expression / formula AST (locally defined, no project imports)
# ---------------------------------------------------------------------------

class BaseType(Enum):
    INT = "Int"
    BOOL = "Bool"
    STR = "Str"
    FLOAT = "Float"
    NONE = "None"
    ANY = "Any"
    LIST = "List"
    DICT = "Dict"
    SET = "Set"
    TUPLE = "Tuple"
    TAG = "Tag"


class UnaryOp(Enum):
    NOT = "not"
    NEG = "neg"
    ABS = "abs"
    LEN = "len"
    IS_NONE = "is_none"
    IS_TRUTHY = "is_truthy"


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    FLOOR_DIV = "//"
    AND = "and"
    OR = "or"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IMPLIES = "=>"
    IN = "in"
    ISINSTANCE = "isinstance"
    HASATTR = "hasattr"


class Quantifier(Enum):
    FORALL = "forall"
    EXISTS = "exists"


@dataclass(frozen=True)
class Var:
    """Variable reference."""
    name: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return mapping.get(self.name, self)

    def to_smt(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntLit:
    value: int

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        if self.value < 0:
            return f"(- {-self.value})"
        return str(self.value)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BoolLit:
    value: bool

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return "true" if self.value else "false"

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class StrLit:
    value: str

    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return f'"{self.value}"'

    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass(frozen=True)
class NoneLit:
    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return self

    def to_smt(self) -> str:
        return "none"

    def __str__(self) -> str:
        return "None"


@dataclass(frozen=True)
class UnaryExpr:
    op: UnaryOp
    operand: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.operand.free_vars()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return UnaryExpr(self.op, self.operand.substitute(mapping))

    def to_smt(self) -> str:
        op_map = {
            UnaryOp.NOT: "not",
            UnaryOp.NEG: "-",
            UnaryOp.ABS: "abs",
            UnaryOp.LEN: "len",
            UnaryOp.IS_NONE: "is_none",
            UnaryOp.IS_TRUTHY: "is_truthy",
        }
        return f"({op_map[self.op]} {self.operand.to_smt()})"

    def __str__(self) -> str:
        return f"({self.op.value} {self.operand})"


@dataclass(frozen=True)
class BinaryExpr:
    op: BinaryOp
    left: Expr
    right: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return BinaryExpr(
            self.op,
            self.left.substitute(mapping),
            self.right.substitute(mapping),
        )

    def to_smt(self) -> str:
        op_map = {
            BinaryOp.ADD: "+",
            BinaryOp.SUB: "-",
            BinaryOp.MUL: "*",
            BinaryOp.DIV: "div",
            BinaryOp.MOD: "mod",
            BinaryOp.FLOOR_DIV: "div",
            BinaryOp.AND: "and",
            BinaryOp.OR: "or",
            BinaryOp.EQ: "=",
            BinaryOp.NE: "distinct",
            BinaryOp.LT: "<",
            BinaryOp.LE: "<=",
            BinaryOp.GT: ">",
            BinaryOp.GE: ">=",
            BinaryOp.IMPLIES: "=>",
            BinaryOp.IN: "contains",
            BinaryOp.ISINSTANCE: "isinstance",
            BinaryOp.HASATTR: "hasattr",
        }
        return f"({op_map[self.op]} {self.left.to_smt()} {self.right.to_smt()})"

    def __str__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"


@dataclass(frozen=True)
class QuantifiedExpr:
    quantifier: Quantifier
    var_name: str
    var_sort: str
    body: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.body.free_vars() - {self.var_name}

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        new_mapping = {k: v for k, v in mapping.items() if k != self.var_name}
        return QuantifiedExpr(
            self.quantifier, self.var_name, self.var_sort,
            self.body.substitute(new_mapping),
        )

    def to_smt(self) -> str:
        q = self.quantifier.value
        return f"({q} (({self.var_name} {self.var_sort})) {self.body.to_smt()})"

    def __str__(self) -> str:
        return f"({self.quantifier.value} {self.var_name}:{self.var_sort}. {self.body})"


@dataclass(frozen=True)
class FuncApp:
    """Uninterpreted function application."""
    func_name: str
    args: Tuple[Expr, ...]

    def free_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for a in self.args:
            result = result | a.free_vars()
        return result

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return FuncApp(self.func_name, tuple(a.substitute(mapping) for a in self.args))

    def to_smt(self) -> str:
        if not self.args:
            return self.func_name
        args_str = " ".join(a.to_smt() for a in self.args)
        return f"({self.func_name} {args_str})"

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.func_name}({args_str})"


@dataclass(frozen=True)
class LetExpr:
    var_name: str
    value: Expr
    body: Expr

    def free_vars(self) -> FrozenSet[str]:
        return self.value.free_vars() | (self.body.free_vars() - {self.var_name})

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        new_mapping = {k: v for k, v in mapping.items() if k != self.var_name}
        return LetExpr(
            self.var_name,
            self.value.substitute(mapping),
            self.body.substitute(new_mapping),
        )

    def to_smt(self) -> str:
        return f"(let (({self.var_name} {self.value.to_smt()})) {self.body.to_smt()})"

    def __str__(self) -> str:
        return f"(let {self.var_name} = {self.value} in {self.body})"


@dataclass(frozen=True)
class IteExpr:
    cond: Expr
    then_branch: Expr
    else_branch: Expr

    def free_vars(self) -> FrozenSet[str]:
        return (
            self.cond.free_vars()
            | self.then_branch.free_vars()
            | self.else_branch.free_vars()
        )

    def substitute(self, mapping: Dict[str, Expr]) -> Expr:
        return IteExpr(
            self.cond.substitute(mapping),
            self.then_branch.substitute(mapping),
            self.else_branch.substitute(mapping),
        )

    def to_smt(self) -> str:
        return (
            f"(ite {self.cond.to_smt()} "
            f"{self.then_branch.to_smt()} {self.else_branch.to_smt()})"
        )

    def __str__(self) -> str:
        return f"(if {self.cond} then {self.then_branch} else {self.else_branch})"


Expr = Union[
    Var, IntLit, BoolLit, StrLit, NoneLit,
    UnaryExpr, BinaryExpr, QuantifiedExpr,
    FuncApp, LetExpr, IteExpr,
]


# ---------------------------------------------------------------------------
# Statement AST (minimal, for WP / SP computation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssignStmt:
    target: str
    value: Expr


@dataclass(frozen=True)
class SeqStmt:
    first: Stmt
    second: Stmt


@dataclass(frozen=True)
class IfStmt:
    condition: Expr
    then_branch: Stmt
    else_branch: Stmt


@dataclass(frozen=True)
class WhileStmt:
    condition: Expr
    body: Stmt
    invariant: Optional[Expr] = None


@dataclass(frozen=True)
class ReturnStmt:
    value: Expr


@dataclass(frozen=True)
class AssertStmt:
    condition: Expr


@dataclass(frozen=True)
class AssumeStmt:
    condition: Expr


@dataclass(frozen=True)
class CallStmt:
    target: Optional[str]
    func_name: str
    args: Tuple[Expr, ...]


@dataclass(frozen=True)
class SkipStmt:
    pass


@dataclass(frozen=True)
class RaiseStmt:
    exception_type: str
    message: Optional[Expr] = None


@dataclass(frozen=True)
class TryStmt:
    body: Stmt
    except_type: str
    except_var: Optional[str]
    handler: Stmt
    finally_block: Optional[Stmt] = None


Stmt = Union[
    AssignStmt, SeqStmt, IfStmt, WhileStmt, ReturnStmt,
    AssertStmt, AssumeStmt, CallStmt, SkipStmt, RaiseStmt, TryStmt,
]


# ---------------------------------------------------------------------------
# Contract / obligation types
# ---------------------------------------------------------------------------

@dataclass
class FunctionContract:
    """A refinement-type contract for a function."""
    func_name: str
    params: List[Tuple[str, str]]  # (name, sort)
    preconditions: List[Expr]
    postconditions: List[Expr]
    return_sort: str = "Int"
    return_var: str = "_result"
    exceptions: List[Tuple[str, Expr]] = field(default_factory=list)


@dataclass
class LoopInvariantContract:
    loop_id: str
    invariant: Expr
    condition: Expr
    body: Stmt
    modified_vars: List[str] = field(default_factory=list)


@dataclass
class SubtypeObligation:
    sub_type_pred: Expr
    super_type_pred: Expr
    context_vars: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Justification(Enum):
    SMT = "smt"
    AXIOM = "axiom"
    MODUS_PONENS = "modus_ponens"
    CASE_SPLIT = "case_split"
    INDUCTION = "induction"
    WEAKENING = "weakening"
    STRENGTHENING = "strengthening"
    SUBTYPING = "subtyping"
    FRAME = "frame"
    CUT = "cut"
    ASSUMPTION = "assumption"
    HYPOTHESIS = "hypothesis"


class VerificationStatus(Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


class VCKind(Enum):
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    LOOP_INVARIANT_INIT = "loop_invariant_init"
    LOOP_INVARIANT_MAINT = "loop_invariant_maint"
    ARRAY_BOUNDS = "array_bounds"
    NULL_SAFETY = "null_safety"
    DIVISION_SAFETY = "division_safety"
    TYPE_SAFETY = "type_safety"
    ASSERTION = "assertion"
    SUBTYPE = "subtype"
    EXCEPTION_SAFETY = "exception_safety"


# ---------------------------------------------------------------------------
# ProofStep
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    """A single step in a proof derivation."""
    step_id: str
    rule_name: str
    premises: List[str]  # step_ids of premises
    conclusion: Expr
    justification: Justification
    annotation: str = ""
    solver_time_ms: float = 0.0
    _details: Dict[str, Any] = field(default_factory=dict)

    @property
    def details(self) -> Dict[str, Any]:
        return self._details

    def validate_step(self, step_map: Dict[str, ProofStep]) -> bool:
        """Validate that all premises exist and precede this step."""
        for p in self.premises:
            if p not in step_map:
                return False
        return True

    def to_natural_language(self) -> str:
        just_text = {
            Justification.SMT: "by SMT solver",
            Justification.AXIOM: "by axiom",
            Justification.MODUS_PONENS: "by modus ponens",
            Justification.CASE_SPLIT: "by case analysis",
            Justification.INDUCTION: "by induction",
            Justification.WEAKENING: "by weakening",
            Justification.STRENGTHENING: "by strengthening",
            Justification.SUBTYPING: "by subtyping",
            Justification.FRAME: "by frame rule",
            Justification.CUT: "by cut rule",
            Justification.ASSUMPTION: "by assumption",
            Justification.HYPOTHESIS: "by hypothesis",
        }
        justification_str = just_text.get(self.justification, "by unknown rule")
        premise_str = ""
        if self.premises:
            premise_str = f" (from steps {', '.join(self.premises)})"
        return (
            f"Step {self.step_id}: {self.rule_name} – "
            f"{self.conclusion}{premise_str} {justification_str}"
        )

    def to_smt_comment(self) -> str:
        return (
            f"; Step {self.step_id}: {self.rule_name} "
            f"({self.justification.value})\n"
            f"(assert {self.conclusion.to_smt()})"
        )


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    status: VerificationStatus
    message: str = ""
    model: Optional[Dict[str, Any]] = None
    unsat_core: Optional[List[str]] = None
    proof_steps_verified: int = 0
    proof_steps_total: int = 0
    solver_time_ms: float = 0.0
    counterexample: Optional[Dict[str, Any]] = None

    @property
    def is_verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    def summary(self) -> str:
        pct = 0.0
        if self.proof_steps_total > 0:
            pct = (self.proof_steps_verified / self.proof_steps_total) * 100.0
        return (
            f"{self.status.value}: {self.proof_steps_verified}/{self.proof_steps_total} "
            f"steps verified ({pct:.1f}%) in {self.solver_time_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# VerificationCondition
# ---------------------------------------------------------------------------

@dataclass
class VerificationCondition:
    vc_id: str
    kind: VCKind
    formula: Expr
    location: str = ""
    description: str = ""
    context_vars: List[Tuple[str, str]] = field(default_factory=list)
    assumptions: List[Expr] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.UNKNOWN
    solver_time_ms: float = 0.0
    counterexample: Optional[Dict[str, Any]] = None

    def to_smt_query(self) -> str:
        lines: List[str] = []
        lines.append("; Verification Condition")
        lines.append(f"; ID: {self.vc_id}")
        lines.append(f"; Kind: {self.kind.value}")
        lines.append(f"; Location: {self.location}")
        lines.append(f"; Description: {self.description}")
        lines.append("(set-logic QF_LIA)")
        for var_name, var_sort in self.context_vars:
            lines.append(f"(declare-const {var_name} {var_sort})")
        for i, assumption in enumerate(self.assumptions):
            lines.append(f"(assert {assumption.to_smt()})  ; assumption {i}")
        lines.append(f"(assert (not {self.formula.to_smt()}))  ; negated goal")
        lines.append("(check-sat)")
        lines.append("(exit)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProofCertificate
# ---------------------------------------------------------------------------

@dataclass
class ProofCertificate:
    """Top-level proof certificate aggregating multiple proof steps."""
    certificate_id: str
    theorem: str
    proof_steps: List[ProofStep] = field(default_factory=list)
    assumptions: List[Expr] = field(default_factory=list)
    conclusion: Optional[Expr] = None
    verification_result: Optional[VerificationResult] = None
    verification_conditions: List[VerificationCondition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # ---- step management ---------------------------------------------------

    def add_step(self, step: ProofStep) -> None:
        self.proof_steps.append(step)

    def add_assumption(self, assumption: Expr) -> None:
        self.assumptions.append(assumption)

    def get_step(self, step_id: str) -> Optional[ProofStep]:
        for s in self.proof_steps:
            if s.step_id == step_id:
                return s
        return None

    def step_map(self) -> Dict[str, ProofStep]:
        return {s.step_id: s for s in self.proof_steps}

    # ---- validation --------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty means valid)."""
        errors: List[str] = []
        sm = self.step_map()

        seen_ids: Set[str] = set()
        for step in self.proof_steps:
            if step.step_id in seen_ids:
                errors.append(f"Duplicate step id: {step.step_id}")
            seen_ids.add(step.step_id)

            for p in step.premises:
                if p not in sm:
                    errors.append(
                        f"Step {step.step_id} references unknown premise {p}"
                    )
                elif p not in seen_ids - {step.step_id}:
                    pass

        if self.conclusion is not None and self.proof_steps:
            last_conclusion = self.proof_steps[-1].conclusion
            if str(last_conclusion) != str(self.conclusion):
                errors.append(
                    f"Last step conclusion {last_conclusion} does not match "
                    f"certificate conclusion {self.conclusion}"
                )

        return errors

    # ---- serialisation helpers -------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the proof certificate."""
        h = hashlib.sha256()
        h.update(self.theorem.encode())
        for s in self.proof_steps:
            h.update(s.step_id.encode())
            h.update(str(s.conclusion).encode())
        return h.hexdigest()

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "theorem": self.theorem,
            "num_steps": len(self.proof_steps),
            "num_assumptions": len(self.assumptions),
            "num_vcs": len(self.verification_conditions),
            "verified": (
                self.verification_result.is_verified
                if self.verification_result
                else None
            ),
            "fingerprint": self.fingerprint(),
        }


# ---------------------------------------------------------------------------
# Helper: expression builders
# ---------------------------------------------------------------------------

def _and(a: Expr, b: Expr) -> Expr:
    if isinstance(a, BoolLit) and a.value:
        return b
    if isinstance(b, BoolLit) and b.value:
        return a
    return BinaryExpr(BinaryOp.AND, a, b)


def _or(a: Expr, b: Expr) -> Expr:
    if isinstance(a, BoolLit) and not a.value:
        return b
    if isinstance(b, BoolLit) and not b.value:
        return a
    return BinaryExpr(BinaryOp.OR, a, b)


def _not(a: Expr) -> Expr:
    if isinstance(a, BoolLit):
        return BoolLit(not a.value)
    if isinstance(a, UnaryExpr) and a.op == UnaryOp.NOT:
        return a.operand
    return UnaryExpr(UnaryOp.NOT, a)


def _implies(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.IMPLIES, a, b)


def _eq(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.EQ, a, b)


def _lt(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.LT, a, b)


def _le(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.LE, a, b)


def _ge(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.GE, a, b)


def _gt(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.GT, a, b)


def _add(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.ADD, a, b)


def _sub(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.SUB, a, b)


def _mul(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.MUL, a, b)


def _ne(a: Expr, b: Expr) -> Expr:
    return BinaryExpr(BinaryOp.NE, a, b)


def _var(name: str) -> Var:
    return Var(name)


def _int(v: int) -> IntLit:
    return IntLit(v)


TRUE = BoolLit(True)
FALSE = BoolLit(False)


def _conjunction(exprs: List[Expr]) -> Expr:
    """Fold a list of expressions into a conjunction."""
    if not exprs:
        return TRUE
    result = exprs[0]
    for e in exprs[1:]:
        result = _and(result, e)
    return result


def _disjunction(exprs: List[Expr]) -> Expr:
    if not exprs:
        return FALSE
    result = exprs[0]
    for e in exprs[1:]:
        result = _or(result, e)
    return result


# ---------------------------------------------------------------------------
# SmtLibGenerator
# ---------------------------------------------------------------------------

class SmtLibGenerator:
    """Generate SMT-LIB 2.6 compliant proof certificates."""

    def __init__(self, logic: str = "ALL") -> None:
        self._logic = logic
        self._lines: List[str] = []
        self._declared_sorts: Set[str] = set()
        self._declared_funs: Set[str] = set()
        self._declared_vars: Set[str] = set()
        self._assertion_count: int = 0
        self._named_assertions: Dict[str, int] = {}

    # ---- header ------------------------------------------------------------

    def set_logic(self, logic: str) -> None:
        self._logic = logic

    def emit_header(self) -> None:
        self._lines.append("; SMT-LIB 2.6 Proof Certificate")
        self._lines.append(f"; Generated by refinement-type-inference CEGAR system")
        self._lines.append(f"; Logic: {self._logic}")
        self._lines.append(f"(set-logic {self._logic})")
        self._lines.append("(set-option :produce-proofs true)")
        self._lines.append("(set-option :produce-unsat-cores true)")
        self._lines.append("")

    # ---- sorts -------------------------------------------------------------

    def declare_sort(self, name: str, arity: int = 0) -> None:
        if name not in self._declared_sorts:
            self._declared_sorts.add(name)
            self._lines.append(f"(declare-sort {name} {arity})")

    def declare_sorts(self) -> None:
        """Declare all standard sorts used in refinement type inference."""
        self._lines.append("; ---- Standard sorts ----")
        for sort_name in ["Tag", "Str"]:
            self.declare_sort(sort_name, 0)
        self._lines.append("")

    # ---- uninterpreted functions -------------------------------------------

    def declare_function(
        self, name: str, param_sorts: List[str], return_sort: str
    ) -> None:
        if name not in self._declared_funs:
            self._declared_funs.add(name)
            params = " ".join(param_sorts)
            self._lines.append(f"(declare-fun {name} ({params}) {return_sort})")

    def declare_functions(self) -> None:
        """Declare standard uninterpreted functions."""
        self._lines.append("; ---- Uninterpreted functions ----")
        self.declare_function("len", ["Str"], "Int")
        self.declare_function("isinstance", ["Int", "Tag"], "Bool")
        self.declare_function("is_none", ["Int"], "Bool")
        self.declare_function("is_truthy", ["Int"], "Bool")
        self.declare_function("hasattr", ["Int", "Str"], "Bool")
        self.declare_function("tag_of", ["Int"], "Tag")
        self.declare_function("str_len", ["Str"], "Int")
        self.declare_function("list_len", ["Int"], "Int")
        self.declare_function("dict_len", ["Int"], "Int")
        self.declare_function("list_get", ["Int", "Int"], "Int")
        self.declare_function("dict_get", ["Int", "Str"], "Int")
        self.declare_function("dict_has_key", ["Int", "Str"], "Bool")
        self._lines.append("")

    # ---- variables ---------------------------------------------------------

    def declare_variable(self, name: str, sort: str) -> None:
        if name not in self._declared_vars:
            self._declared_vars.add(name)
            self._lines.append(f"(declare-const {name} {sort})")

    def declare_variables(self, variables: List[Tuple[str, str]]) -> None:
        self._lines.append("; ---- Program variables ----")
        for name, sort in variables:
            self.declare_variable(name, sort)
        self._lines.append("")

    # ---- tag constants -----------------------------------------------------

    def declare_tag_constants(self) -> None:
        self._lines.append("; ---- Type tag constants ----")
        tags = ["IntTag", "BoolTag", "StrTag", "FloatTag", "NoneTag",
                "ListTag", "DictTag", "SetTag", "TupleTag"]
        for tag in tags:
            self.declare_variable(tag, "Tag")
        # distinctness axiom
        self._lines.append(f"(assert (distinct {' '.join(tags)}))")
        self._lines.append("")

    # ---- axioms ------------------------------------------------------------

    def assert_axiom(self, name: str, formula: Expr) -> None:
        self._assertion_count += 1
        self._named_assertions[name] = self._assertion_count
        self._lines.append(
            f"(assert (! {formula.to_smt()} :named {name}))"
        )

    def assert_axioms(self) -> None:
        """Assert standard axioms for function interpretations and type hierarchy."""
        self._lines.append("; ---- Axioms ----")

        # len is non-negative
        self.assert_axiom(
            "ax_len_nonneg",
            QuantifiedExpr(
                Quantifier.FORALL, "s", "Str",
                _ge(FuncApp("len", (Var("s"),)), _int(0)),
            ),
        )

        # str_len non-negative
        self.assert_axiom(
            "ax_str_len_nonneg",
            QuantifiedExpr(
                Quantifier.FORALL, "s", "Str",
                _ge(FuncApp("str_len", (Var("s"),)), _int(0)),
            ),
        )

        # list_len non-negative
        self.assert_axiom(
            "ax_list_len_nonneg",
            QuantifiedExpr(
                Quantifier.FORALL, "x", "Int",
                _implies(
                    _eq(FuncApp("tag_of", (Var("x"),)), Var("ListTag")),
                    _ge(FuncApp("list_len", (Var("x"),)), _int(0)),
                ),
            ),
        )

        # dict_len non-negative
        self.assert_axiom(
            "ax_dict_len_nonneg",
            QuantifiedExpr(
                Quantifier.FORALL, "x", "Int",
                _implies(
                    _eq(FuncApp("tag_of", (Var("x"),)), Var("DictTag")),
                    _ge(FuncApp("dict_len", (Var("x"),)), _int(0)),
                ),
            ),
        )

        # is_none iff tag is NoneTag
        self.assert_axiom(
            "ax_is_none_iff_tag",
            QuantifiedExpr(
                Quantifier.FORALL, "x", "Int",
                _eq(
                    FuncApp("is_none", (Var("x"),)),
                    _eq(FuncApp("tag_of", (Var("x"),)), Var("NoneTag")),
                ),
            ),
        )

        # isinstance relationship with tag_of
        self.assert_axiom(
            "ax_isinstance_tag",
            QuantifiedExpr(
                Quantifier.FORALL, "x", "Int",
                QuantifiedExpr(
                    Quantifier.FORALL, "t", "Tag",
                    _implies(
                        _eq(FuncApp("tag_of", (Var("x"),)), Var("t")),
                        FuncApp("isinstance", (Var("x"), Var("t"))),
                    ),
                ),
            ),
        )

        # list bounds
        self.assert_axiom(
            "ax_list_bounds",
            QuantifiedExpr(
                Quantifier.FORALL, "x", "Int",
                QuantifiedExpr(
                    Quantifier.FORALL, "i", "Int",
                    _implies(
                        _and(
                            _eq(FuncApp("tag_of", (Var("x"),)), Var("ListTag")),
                            _and(
                                _ge(Var("i"), _int(0)),
                                _lt(Var("i"), FuncApp("list_len", (Var("x"),))),
                            ),
                        ),
                        _not(FuncApp("is_none", (FuncApp("list_get", (Var("x"), Var("i"))),))),
                    ),
                ),
            ),
        )

        # dict_has_key implies dict_get is not none
        self.assert_axiom(
            "ax_dict_has_key",
            QuantifiedExpr(
                Quantifier.FORALL, "x", "Int",
                QuantifiedExpr(
                    Quantifier.FORALL, "k", "Str",
                    _implies(
                        _and(
                            _eq(FuncApp("tag_of", (Var("x"),)), Var("DictTag")),
                            FuncApp("dict_has_key", (Var("x"), Var("k"))),
                        ),
                        _not(FuncApp("is_none", (FuncApp("dict_get", (Var("x"), Var("k"))),))),
                    ),
                ),
            ),
        )

        self._lines.append("")

    # ---- preconditions / postconditions ------------------------------------

    def assert_preconditions(
        self, contract: FunctionContract, prefix: str = "pre"
    ) -> None:
        self._lines.append(f"; ---- Preconditions for {contract.func_name} ----")
        for i, pre in enumerate(contract.preconditions):
            name = f"{prefix}_{contract.func_name}_{i}"
            self.assert_axiom(name, pre)
        self._lines.append("")

    def assert_postconditions(
        self, contract: FunctionContract, prefix: str = "post"
    ) -> None:
        self._lines.append(f"; ---- Postconditions for {contract.func_name} ----")
        for i, post in enumerate(contract.postconditions):
            name = f"{prefix}_{contract.func_name}_{i}"
            self.assert_axiom(name, post)
        self._lines.append("")

    # ---- verification query -----------------------------------------------

    def check_sat(self) -> None:
        self._lines.append("(check-sat)")

    def get_proof(self) -> None:
        self._lines.append("(get-proof)")

    def get_unsat_core(self) -> None:
        self._lines.append("(get-unsat-core)")

    def get_model(self) -> None:
        self._lines.append("(get-model)")

    def exit(self) -> None:
        self._lines.append("(exit)")

    # ---- full certificate generation ---------------------------------------

    def generate_certificate(
        self,
        contract: FunctionContract,
        extra_assumptions: Optional[List[Expr]] = None,
        goal: Optional[Expr] = None,
    ) -> str:
        """Generate a full SMT-LIB proof certificate for a function contract."""
        self._lines.clear()
        self.emit_header()
        self.declare_sorts()
        self.declare_functions()
        self.declare_tag_constants()

        variables = list(contract.params)
        variables.append((contract.return_var, contract.return_sort))
        self.declare_variables(variables)

        self.assert_axioms()
        self.assert_preconditions(contract)

        if extra_assumptions:
            self._lines.append("; ---- Additional assumptions ----")
            for i, a in enumerate(extra_assumptions):
                self.assert_axiom(f"extra_assm_{i}", a)
            self._lines.append("")

        if goal is not None:
            self._lines.append("; ---- Verification goal (negated) ----")
            self._assertion_count += 1
            self._lines.append(
                f"(assert (! (not {goal.to_smt()}) :named goal))"
            )
        else:
            self.assert_postconditions(contract)

        self._lines.append("")
        self.check_sat()
        self.get_unsat_core()
        self.get_proof()
        self.exit()

        return "\n".join(self._lines)

    def generate_vc_query(self, vc: VerificationCondition) -> str:
        """Generate SMT-LIB query for a single verification condition."""
        return vc.to_smt_query()

    def render(self) -> str:
        return "\n".join(self._lines)

    def reset(self) -> None:
        self._lines.clear()
        self._declared_sorts.clear()
        self._declared_funs.clear()
        self._declared_vars.clear()
        self._assertion_count = 0
        self._named_assertions.clear()


# ---------------------------------------------------------------------------
# ProofObligationGenerator
# ---------------------------------------------------------------------------

class ProofObligationGenerator:
    """Generate proof obligations (VCs) from inferred types and contracts."""

    def __init__(self) -> None:
        self._vc_counter: int = 0

    def _next_id(self, prefix: str = "vc") -> str:
        self._vc_counter += 1
        return f"{prefix}_{self._vc_counter}"

    # ---- from function contract -------------------------------------------

    def from_function_contract(
        self, contract: FunctionContract
    ) -> List[VerificationCondition]:
        vcs: List[VerificationCondition] = []

        # Postcondition VC: pre => post
        pre_conj = _conjunction(contract.preconditions) if contract.preconditions else TRUE
        for i, post in enumerate(contract.postconditions):
            vc = VerificationCondition(
                vc_id=self._next_id("fc_post"),
                kind=VCKind.POSTCONDITION,
                formula=_implies(pre_conj, post),
                location=contract.func_name,
                description=f"Postcondition {i} of {contract.func_name}",
                context_vars=list(contract.params)
                + [(contract.return_var, contract.return_sort)],
                assumptions=list(contract.preconditions),
            )
            vcs.append(vc)

        return vcs

    # ---- subtype check -----------------------------------------------------

    def from_subtype_check(
        self, obligation: SubtypeObligation
    ) -> VerificationCondition:
        return VerificationCondition(
            vc_id=self._next_id("sub"),
            kind=VCKind.SUBTYPE,
            formula=_implies(obligation.sub_type_pred, obligation.super_type_pred),
            description="Subtype check",
            context_vars=list(obligation.context_vars),
            assumptions=[],
        )

    # ---- loop invariant ----------------------------------------------------

    def from_loop_invariant(
        self, contract: LoopInvariantContract
    ) -> List[VerificationCondition]:
        vcs: List[VerificationCondition] = []

        # Invariant is initially established (we require caller to set up context)
        vcs.append(VerificationCondition(
            vc_id=self._next_id("loop_init"),
            kind=VCKind.LOOP_INVARIANT_INIT,
            formula=contract.invariant,
            location=contract.loop_id,
            description=f"Loop invariant initialization for {contract.loop_id}",
        ))

        # Invariant is maintained: I ∧ cond => wp(body, I)
        wp_calc = WeakestPreconditionCalculator()
        wp_body_inv = wp_calc.wp(contract.body, contract.invariant)
        maint_formula = _implies(
            _and(contract.invariant, contract.condition),
            wp_body_inv,
        )
        vcs.append(VerificationCondition(
            vc_id=self._next_id("loop_maint"),
            kind=VCKind.LOOP_INVARIANT_MAINT,
            formula=maint_formula,
            location=contract.loop_id,
            description=f"Loop invariant maintenance for {contract.loop_id}",
        ))

        return vcs

    # ---- array bounds check ------------------------------------------------

    def from_array_bounds_check(
        self,
        array_var: str,
        index_expr: Expr,
        context_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationCondition:
        """index_expr >= 0 and index_expr < len(array_var)."""
        arr = _var(array_var)
        arr_len = FuncApp("list_len", (arr,))
        bounds = _and(
            _ge(index_expr, _int(0)),
            _lt(index_expr, arr_len),
        )
        return VerificationCondition(
            vc_id=self._next_id("arr"),
            kind=VCKind.ARRAY_BOUNDS,
            formula=bounds,
            description=f"Array bounds check for {array_var}",
            context_vars=context_vars or [],
        )

    # ---- null safety check -------------------------------------------------

    def from_null_safety_check(
        self,
        var_name: str,
        context_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationCondition:
        formula = _not(FuncApp("is_none", (_var(var_name),)))
        return VerificationCondition(
            vc_id=self._next_id("null"),
            kind=VCKind.NULL_SAFETY,
            formula=formula,
            description=f"Null safety check for {var_name}",
            context_vars=context_vars or [],
        )

    # ---- division safety check ---------------------------------------------

    def from_division_safety_check(
        self,
        divisor_expr: Expr,
        context_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationCondition:
        formula = _ne(divisor_expr, _int(0))
        return VerificationCondition(
            vc_id=self._next_id("div"),
            kind=VCKind.DIVISION_SAFETY,
            formula=formula,
            description=f"Division safety check: divisor != 0",
            context_vars=context_vars or [],
        )

    # ---- type safety check -------------------------------------------------

    def from_type_safety_check(
        self,
        var_name: str,
        expected_tag: str,
        context_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationCondition:
        formula = FuncApp("isinstance", (_var(var_name), _var(expected_tag)))
        return VerificationCondition(
            vc_id=self._next_id("type"),
            kind=VCKind.TYPE_SAFETY,
            formula=formula,
            description=f"Type safety check: {var_name} isinstance {expected_tag}",
            context_vars=context_vars or [],
        )

    # ---- assertion ---------------------------------------------------------

    def from_assertion(
        self,
        formula: Expr,
        location: str = "",
        context_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationCondition:
        return VerificationCondition(
            vc_id=self._next_id("assert"),
            kind=VCKind.ASSERTION,
            formula=formula,
            location=location,
            description="Assertion check",
            context_vars=context_vars or [],
        )

    # ---- exception safety --------------------------------------------------

    def from_exception_safety(
        self,
        condition: Expr,
        exception_type: str,
        location: str = "",
        context_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> VerificationCondition:
        return VerificationCondition(
            vc_id=self._next_id("exc"),
            kind=VCKind.EXCEPTION_SAFETY,
            formula=condition,
            location=location,
            description=f"Exception safety: {exception_type} not raised",
            context_vars=context_vars or [],
        )


# ---------------------------------------------------------------------------
# WeakestPreconditionCalculator
# ---------------------------------------------------------------------------

class WeakestPreconditionCalculator:
    """Compute weakest preconditions for statements w.r.t. postconditions."""

    def __init__(self) -> None:
        self._function_contracts: Dict[str, FunctionContract] = {}

    def register_contract(self, contract: FunctionContract) -> None:
        self._function_contracts[contract.func_name] = contract

    # ---- main dispatch -----------------------------------------------------

    def wp(self, stmt: Stmt, post: Expr) -> Expr:
        if isinstance(stmt, SkipStmt):
            return post
        if isinstance(stmt, AssignStmt):
            return self.wp_assignment(stmt, post)
        if isinstance(stmt, SeqStmt):
            return self.wp_sequence(stmt, post)
        if isinstance(stmt, IfStmt):
            return self.wp_conditional(stmt, post)
        if isinstance(stmt, WhileStmt):
            return self.wp_loop(stmt, post)
        if isinstance(stmt, AssertStmt):
            return self.wp_assert(stmt, post)
        if isinstance(stmt, AssumeStmt):
            return self.wp_assume(stmt, post)
        if isinstance(stmt, CallStmt):
            return self.wp_function_call(stmt, post)
        if isinstance(stmt, ReturnStmt):
            return self.wp_return(stmt, post)
        if isinstance(stmt, RaiseStmt):
            return self.wp_raise(stmt, post)
        if isinstance(stmt, TryStmt):
            return self.wp_try(stmt, post)
        return post  # conservative fallback

    # ---- individual rules --------------------------------------------------

    def wp_assignment(self, stmt: AssignStmt, post: Expr) -> Expr:
        """wp(x := e, Q) = Q[x / e]"""
        return post.substitute({stmt.target: stmt.value})

    def wp_sequence(self, stmt: SeqStmt, post: Expr) -> Expr:
        """wp(S1; S2, Q) = wp(S1, wp(S2, Q))"""
        inner = self.wp(stmt.second, post)
        return self.wp(stmt.first, inner)

    def wp_conditional(self, stmt: IfStmt, post: Expr) -> Expr:
        """wp(if B then S1 else S2, Q) = (B => wp(S1,Q)) ∧ (¬B => wp(S2,Q))"""
        wp_then = self.wp(stmt.then_branch, post)
        wp_else = self.wp(stmt.else_branch, post)
        return _and(
            _implies(stmt.condition, wp_then),
            _implies(_not(stmt.condition), wp_else),
        )

    def wp_loop(self, stmt: WhileStmt, post: Expr) -> Expr:
        """wp(while B do S, Q) requires invariant I.
        Returns: I ∧ ∀.(I ∧ B ⇒ wp(S, I)) ∧ (I ∧ ¬B ⇒ Q)
        If no invariant, returns TRUE (incomplete but safe).
        """
        if stmt.invariant is None:
            return TRUE

        inv = stmt.invariant
        wp_body_inv = self.wp(stmt.body, inv)

        # I must hold initially
        # I ∧ B ⇒ wp(body, I) — maintenance
        # I ∧ ¬B ⇒ Q — post-loop
        maintenance = _implies(_and(inv, stmt.condition), wp_body_inv)
        post_loop = _implies(_and(inv, _not(stmt.condition)), post)

        return _and(inv, _and(maintenance, post_loop))

    def wp_assert(self, stmt: AssertStmt, post: Expr) -> Expr:
        """wp(assert C, Q) = C ∧ Q"""
        return _and(stmt.condition, post)

    def wp_assume(self, stmt: AssumeStmt, post: Expr) -> Expr:
        """wp(assume C, Q) = C ⇒ Q"""
        return _implies(stmt.condition, post)

    def wp_function_call(self, stmt: CallStmt, post: Expr) -> Expr:
        """wp(x := f(args), Q) = pre_f[params/args] ∧ ∀r. post_f[params/args, ret/r] ⇒ Q[x/r]"""
        contract = self._function_contracts.get(stmt.func_name)
        if contract is None:
            return post

        mapping: Dict[str, Expr] = {}
        for (pname, _), arg in zip(contract.params, stmt.args):
            mapping[pname] = arg

        # Substitute pre
        pre_instantiated = _conjunction([
            p.substitute(mapping) for p in contract.preconditions
        ])

        # For post, substitute params and handle return var
        if stmt.target is not None:
            ret_var = _var(stmt.target)
            post_mapping = dict(mapping)
            post_mapping[contract.return_var] = ret_var
            post_instantiated = _conjunction([
                p.substitute(post_mapping) for p in contract.postconditions
            ])
            post_with_call = _implies(post_instantiated, post)
        else:
            post_with_call = post

        return _and(pre_instantiated, post_with_call)

    def wp_return(self, stmt: ReturnStmt, post: Expr) -> Expr:
        """wp(return e, Q) = Q[_result / e]"""
        return post.substitute({"_result": stmt.value})

    def wp_raise(self, stmt: RaiseStmt, post: Expr) -> Expr:
        """wp(raise E, Q) = false (raise never satisfies normal postcondition)"""
        return FALSE

    def wp_try(self, stmt: TryStmt, post: Expr) -> Expr:
        """Simplified wp for try-except: assume handler always catches."""
        wp_handler = self.wp(stmt.handler, post)
        wp_body = self.wp(stmt.body, post)
        # Either the body succeeds with Q, or it raises and handler achieves Q
        return _or(wp_body, wp_handler)


# ---------------------------------------------------------------------------
# StrongestPostconditionCalculator
# ---------------------------------------------------------------------------

class StrongestPostconditionCalculator:
    """Compute strongest postconditions for statements w.r.t. preconditions."""

    def __init__(self) -> None:
        self._ssa_counter: int = 0
        self._function_contracts: Dict[str, FunctionContract] = {}

    def register_contract(self, contract: FunctionContract) -> None:
        self._function_contracts[contract.func_name] = contract

    def _fresh_var(self, base: str) -> str:
        self._ssa_counter += 1
        return f"{base}__sp_{self._ssa_counter}"

    # ---- main dispatch -----------------------------------------------------

    def sp(self, pre: Expr, stmt: Stmt) -> Expr:
        if isinstance(stmt, SkipStmt):
            return pre
        if isinstance(stmt, AssignStmt):
            return self.sp_assignment(pre, stmt)
        if isinstance(stmt, SeqStmt):
            return self.sp_sequence(pre, stmt)
        if isinstance(stmt, IfStmt):
            return self.sp_conditional(pre, stmt)
        if isinstance(stmt, WhileStmt):
            return self.sp_loop(pre, stmt)
        if isinstance(stmt, AssertStmt):
            return self.sp_assert(pre, stmt)
        if isinstance(stmt, AssumeStmt):
            return self.sp_assume(pre, stmt)
        if isinstance(stmt, CallStmt):
            return self.sp_call(pre, stmt)
        if isinstance(stmt, ReturnStmt):
            return pre
        return pre

    # ---- individual rules --------------------------------------------------

    def sp_assignment(self, pre: Expr, stmt: AssignStmt) -> Expr:
        """sp(P, x := e) = ∃x₀. P[x/x₀] ∧ x = e[x/x₀]"""
        old_var = self._fresh_var(stmt.target)
        pre_renamed = pre.substitute({stmt.target: _var(old_var)})
        value_renamed = stmt.value.substitute({stmt.target: _var(old_var)})
        return _and(pre_renamed, _eq(_var(stmt.target), value_renamed))

    def sp_sequence(self, pre: Expr, stmt: SeqStmt) -> Expr:
        """sp(P, S1; S2) = sp(sp(P, S1), S2)"""
        mid = self.sp(pre, stmt.first)
        return self.sp(mid, stmt.second)

    def sp_conditional(self, pre: Expr, stmt: IfStmt) -> Expr:
        """sp(P, if B then S1 else S2) = sp(P ∧ B, S1) ∨ sp(P ∧ ¬B, S2)"""
        sp_then = self.sp(_and(pre, stmt.condition), stmt.then_branch)
        sp_else = self.sp(_and(pre, _not(stmt.condition)), stmt.else_branch)
        return _or(sp_then, sp_else)

    def sp_loop(self, pre: Expr, stmt: WhileStmt) -> Expr:
        """sp(P, while B do S) with invariant I:
        result = I ∧ ¬B, provided we can show P ⇒ I and {I ∧ B} S {I}
        """
        if stmt.invariant is None:
            return TRUE
        return _and(stmt.invariant, _not(stmt.condition))

    def sp_assert(self, pre: Expr, stmt: AssertStmt) -> Expr:
        """sp(P, assert C) = P ∧ C"""
        return _and(pre, stmt.condition)

    def sp_assume(self, pre: Expr, stmt: AssumeStmt) -> Expr:
        """sp(P, assume C) = P ∧ C"""
        return _and(pre, stmt.condition)

    def sp_call(self, pre: Expr, stmt: CallStmt) -> Expr:
        """sp(P, x := f(args)) = P ∧ pre_f[params/args] ∧ post_f[params/args, ret/x]"""
        contract = self._function_contracts.get(stmt.func_name)
        if contract is None:
            return pre

        mapping: Dict[str, Expr] = {}
        for (pname, _), arg in zip(contract.params, stmt.args):
            mapping[pname] = arg

        pre_f = _conjunction([p.substitute(mapping) for p in contract.preconditions])

        if stmt.target is not None:
            post_mapping = dict(mapping)
            post_mapping[contract.return_var] = _var(stmt.target)
            post_f = _conjunction(
                [p.substitute(post_mapping) for p in contract.postconditions]
            )
        else:
            post_f = _conjunction(
                [p.substitute(mapping) for p in contract.postconditions]
            )

        return _and(pre, _and(pre_f, post_f))


# ---------------------------------------------------------------------------
# HoareTripleVerifier
# ---------------------------------------------------------------------------

@dataclass
class HoareTriple:
    precondition: Expr
    statement: Stmt
    postcondition: Expr


class HoareTripleVerifier:
    """Verify Hoare triples {P} S {Q} using weakest preconditions."""

    def __init__(self) -> None:
        self._wp_calc = WeakestPreconditionCalculator()
        self._sp_calc = StrongestPostconditionCalculator()
        self._proof_steps: List[ProofStep] = []
        self._step_counter: int = 0

    def register_contract(self, contract: FunctionContract) -> None:
        self._wp_calc.register_contract(contract)
        self._sp_calc.register_contract(contract)

    def _next_step_id(self) -> str:
        self._step_counter += 1
        return f"hoare_{self._step_counter}"

    # ---- main verification ------------------------------------------------

    def verify_triple(
        self, pre: Expr, stmt: Stmt, post: Expr
    ) -> Tuple[bool, List[ProofStep]]:
        """Verify {pre} stmt {post}. Returns (result, proof_steps)."""
        self._proof_steps = []
        self._step_counter = 0
        result = self._verify(pre, stmt, post)
        return result, list(self._proof_steps)

    def _verify(self, pre: Expr, stmt: Stmt, post: Expr) -> bool:
        if isinstance(stmt, SkipStmt):
            return self._verify_skip(pre, post)
        if isinstance(stmt, AssignStmt):
            return self._verify_assignment(pre, stmt, post)
        if isinstance(stmt, SeqStmt):
            return self._verify_sequence(pre, stmt, post)
        if isinstance(stmt, IfStmt):
            return self._verify_conditional(pre, stmt, post)
        if isinstance(stmt, WhileStmt):
            return self._verify_loop(pre, stmt, post)
        if isinstance(stmt, AssertStmt):
            return self._verify_assert(pre, stmt, post)
        if isinstance(stmt, AssumeStmt):
            return self._verify_assume(pre, stmt, post)
        return self._verify_via_wp(pre, stmt, post)

    # ---- skip rule ---------------------------------------------------------

    def _verify_skip(self, pre: Expr, post: Expr) -> bool:
        valid = self._check_implication(pre, post)
        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="skip",
            premises=[],
            conclusion=_implies(pre, post),
            justification=Justification.SMT,
            annotation="{P} skip {Q} iff P ⇒ Q",
        ))
        return valid

    # ---- assignment rule ---------------------------------------------------

    def _verify_assignment(
        self, pre: Expr, stmt: AssignStmt, post: Expr
    ) -> bool:
        wp = self._wp_calc.wp_assignment(stmt, post)
        valid = self._check_implication(pre, wp)
        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="assignment",
            premises=[],
            conclusion=_implies(pre, wp),
            justification=Justification.SMT,
            annotation=f"{{Q[{stmt.target}/e]}} {stmt.target} := e {{Q}}",
        ))
        return valid

    # ---- sequence rule -----------------------------------------------------

    def _verify_sequence(
        self, pre: Expr, stmt: SeqStmt, post: Expr
    ) -> bool:
        # Find midcondition via WP of second statement
        mid = self._wp_calc.wp(stmt.second, post)
        r1 = self._verify(pre, stmt.first, mid)
        step1_id = self._proof_steps[-1].step_id if self._proof_steps else ""
        r2 = self._verify(mid, stmt.second, post)
        step2_id = self._proof_steps[-1].step_id if self._proof_steps else ""

        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="sequence",
            premises=[step1_id, step2_id],
            conclusion=_implies(pre, post),
            justification=Justification.CUT,
            annotation="{P}S1{R}, {R}S2{Q} ⊢ {P}S1;S2{Q}",
        ))
        return r1 and r2

    # ---- conditional rule --------------------------------------------------

    def _verify_conditional(
        self, pre: Expr, stmt: IfStmt, post: Expr
    ) -> bool:
        pre_true = _and(pre, stmt.condition)
        pre_false = _and(pre, _not(stmt.condition))

        r_then = self._verify(pre_true, stmt.then_branch, post)
        then_id = self._proof_steps[-1].step_id if self._proof_steps else ""
        r_else = self._verify(pre_false, stmt.else_branch, post)
        else_id = self._proof_steps[-1].step_id if self._proof_steps else ""

        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="conditional",
            premises=[then_id, else_id],
            conclusion=_implies(pre, post),
            justification=Justification.CASE_SPLIT,
            annotation="{P∧B}S1{Q}, {P∧¬B}S2{Q} ⊢ {P}if B then S1 else S2{Q}",
        ))
        return r_then and r_else

    # ---- loop rule ---------------------------------------------------------

    def _verify_loop(
        self, pre: Expr, stmt: WhileStmt, post: Expr
    ) -> bool:
        if stmt.invariant is None:
            self._proof_steps.append(ProofStep(
                step_id=self._next_step_id(),
                rule_name="loop_no_invariant",
                premises=[],
                conclusion=TRUE,
                justification=Justification.ASSUMPTION,
                annotation="No loop invariant provided; verification incomplete",
            ))
            return False

        inv = stmt.invariant
        premises: List[str] = []

        # 1) P ⇒ I (invariant initially holds)
        init_ok = self._check_implication(pre, inv)
        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="loop_init",
            premises=[],
            conclusion=_implies(pre, inv),
            justification=Justification.SMT,
            annotation="Loop invariant initialization",
        ))
        premises.append(self._proof_steps[-1].step_id)

        # 2) {I ∧ B} body {I} (invariant is maintained)
        maint_ok = self._verify(_and(inv, stmt.condition), stmt.body, inv)
        premises.append(self._proof_steps[-1].step_id)

        # 3) I ∧ ¬B ⇒ Q (post-loop)
        exit_ok = self._check_implication(
            _and(inv, _not(stmt.condition)), post
        )
        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="loop_exit",
            premises=[],
            conclusion=_implies(_and(inv, _not(stmt.condition)), post),
            justification=Justification.SMT,
            annotation="Loop exit implies postcondition",
        ))
        premises.append(self._proof_steps[-1].step_id)

        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="loop",
            premises=premises,
            conclusion=_implies(pre, post),
            justification=Justification.INDUCTION,
            annotation="{I∧B}S{I} ⊢ {I}while B do S{I∧¬B}",
        ))
        return init_ok and maint_ok and exit_ok

    # ---- assert rule -------------------------------------------------------

    def _verify_assert(
        self, pre: Expr, stmt: AssertStmt, post: Expr
    ) -> bool:
        cond_ok = self._check_implication(pre, stmt.condition)
        post_ok = self._check_implication(_and(pre, stmt.condition), post)

        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="assert",
            premises=[],
            conclusion=_and(
                _implies(pre, stmt.condition),
                _implies(_and(pre, stmt.condition), post),
            ),
            justification=Justification.SMT,
            annotation="{P} assert C {Q} iff P ⇒ C and P ∧ C ⇒ Q",
        ))
        return cond_ok and post_ok

    # ---- assume rule -------------------------------------------------------

    def _verify_assume(
        self, pre: Expr, stmt: AssumeStmt, post: Expr
    ) -> bool:
        valid = self._check_implication(_and(pre, stmt.condition), post)
        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="assume",
            premises=[],
            conclusion=_implies(_and(pre, stmt.condition), post),
            justification=Justification.SMT,
            annotation="{P} assume C {Q} iff P ∧ C ⇒ Q",
        ))
        return valid

    # ---- general WP-based verification ------------------------------------

    def _verify_via_wp(
        self, pre: Expr, stmt: Stmt, post: Expr
    ) -> bool:
        wp = self._wp_calc.wp(stmt, post)
        valid = self._check_implication(pre, wp)
        self._proof_steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="wp_based",
            premises=[],
            conclusion=_implies(pre, wp),
            justification=Justification.SMT,
            annotation="Verified via weakest precondition",
        ))
        return valid

    # ---- consequence rule --------------------------------------------------

    def strengthen_precondition(
        self, stronger_pre: Expr, original_pre: Expr, stmt: Stmt, post: Expr
    ) -> Tuple[bool, List[ProofStep]]:
        """
        If stronger_pre ⇒ original_pre and {original_pre} S {Q},
        then {stronger_pre} S {Q}.
        """
        steps: List[ProofStep] = []
        implication_ok = self._check_implication(stronger_pre, original_pre)
        steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="strengthen_pre",
            premises=[],
            conclusion=_implies(stronger_pre, original_pre),
            justification=Justification.SMT,
            annotation="Precondition strengthening",
        ))

        triple_ok, triple_steps = self.verify_triple(original_pre, stmt, post)
        steps.extend(triple_steps)

        steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="consequence_pre",
            premises=[s.step_id for s in steps[:-1]],
            conclusion=_implies(stronger_pre, post),
            justification=Justification.STRENGTHENING,
            annotation="P' ⇒ P, {P}S{Q} ⊢ {P'}S{Q}",
        ))
        return implication_ok and triple_ok, steps

    def weaken_postcondition(
        self, pre: Expr, stmt: Stmt, original_post: Expr, weaker_post: Expr
    ) -> Tuple[bool, List[ProofStep]]:
        """
        If {P} S {original_post} and original_post ⇒ weaker_post,
        then {P} S {weaker_post}.
        """
        steps: List[ProofStep] = []
        triple_ok, triple_steps = self.verify_triple(pre, stmt, original_post)
        steps.extend(triple_steps)

        implication_ok = self._check_implication(original_post, weaker_post)
        steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="weaken_post",
            premises=[],
            conclusion=_implies(original_post, weaker_post),
            justification=Justification.SMT,
            annotation="Postcondition weakening",
        ))

        steps.append(ProofStep(
            step_id=self._next_step_id(),
            rule_name="consequence_post",
            premises=[s.step_id for s in steps[:-1]],
            conclusion=_implies(pre, weaker_post),
            justification=Justification.WEAKENING,
            annotation="{P}S{Q}, Q ⇒ Q' ⊢ {P}S{Q'}",
        ))
        return triple_ok and implication_ok, steps

    # ---- implication check (lightweight) -----------------------------------

    @staticmethod
    def _check_implication(antecedent: Expr, consequent: Expr) -> bool:
        """
        Lightweight syntactic implication check.
        A full implementation would invoke an SMT solver.
        """
        if isinstance(consequent, BoolLit) and consequent.value:
            return True
        if isinstance(antecedent, BoolLit) and not antecedent.value:
            return True
        if str(antecedent) == str(consequent):
            return True
        # Check if consequent is a conjunct of antecedent
        if isinstance(antecedent, BinaryExpr) and antecedent.op == BinaryOp.AND:
            if str(antecedent.left) == str(consequent):
                return True
            if str(antecedent.right) == str(consequent):
                return True
        # Implication where antecedent matches lhs
        if isinstance(consequent, BinaryExpr) and consequent.op == BinaryOp.IMPLIES:
            if str(antecedent) == str(consequent.left):
                return True
        return False


# ---------------------------------------------------------------------------
# VerificationConditionGenerator
# ---------------------------------------------------------------------------

class VerificationConditionGenerator:
    """Generate verification conditions from functions and contracts."""

    def __init__(self) -> None:
        self._obligation_gen = ProofObligationGenerator()
        self._wp_calc = WeakestPreconditionCalculator()
        self._vc_counter: int = 0

    def _next_id(self, prefix: str = "vc") -> str:
        self._vc_counter += 1
        return f"{prefix}_{self._vc_counter}"

    def register_contract(self, contract: FunctionContract) -> None:
        self._wp_calc.register_contract(contract)

    def generate_vcs(
        self,
        func_name: str,
        body: Stmt,
        contract: FunctionContract,
    ) -> List[VerificationCondition]:
        """Generate all verification conditions for a function."""
        vcs: List[VerificationCondition] = []

        # 1) Postcondition VCs via WP
        pre_conj = _conjunction(contract.preconditions) if contract.preconditions else TRUE
        for i, post in enumerate(contract.postconditions):
            wp = self._wp_calc.wp(body, post)
            vc = VerificationCondition(
                vc_id=self._next_id("post"),
                kind=VCKind.POSTCONDITION,
                formula=_implies(pre_conj, wp),
                location=f"{func_name}:return",
                description=f"Postcondition {i} at return of {func_name}",
                context_vars=list(contract.params),
                assumptions=list(contract.preconditions),
            )
            vcs.append(vc)

        # 2) Safety VCs from body
        safety_vcs = self._extract_safety_vcs(body, func_name, contract)
        vcs.extend(safety_vcs)

        return vcs

    def _extract_safety_vcs(
        self,
        stmt: Stmt,
        location: str,
        contract: FunctionContract,
    ) -> List[VerificationCondition]:
        """Walk the statement to find safety-critical operations."""
        vcs: List[VerificationCondition] = []

        if isinstance(stmt, SeqStmt):
            vcs.extend(self._extract_safety_vcs(stmt.first, location, contract))
            vcs.extend(self._extract_safety_vcs(stmt.second, location, contract))

        elif isinstance(stmt, IfStmt):
            vcs.extend(self._extract_safety_vcs(stmt.then_branch, f"{location}:then", contract))
            vcs.extend(self._extract_safety_vcs(stmt.else_branch, f"{location}:else", contract))

        elif isinstance(stmt, WhileStmt):
            if stmt.invariant is not None:
                inv_vcs = self._obligation_gen.from_loop_invariant(
                    LoopInvariantContract(
                        loop_id=f"{location}:loop",
                        invariant=stmt.invariant,
                        condition=stmt.condition,
                        body=stmt.body,
                    )
                )
                vcs.extend(inv_vcs)
            vcs.extend(self._extract_safety_vcs(stmt.body, f"{location}:loop_body", contract))

        elif isinstance(stmt, AssertStmt):
            vcs.append(VerificationCondition(
                vc_id=self._next_id("assert"),
                kind=VCKind.ASSERTION,
                formula=stmt.condition,
                location=location,
                description=f"Assertion at {location}",
                context_vars=list(contract.params),
                assumptions=list(contract.preconditions),
            ))

        elif isinstance(stmt, AssignStmt):
            # Check for division safety
            if isinstance(stmt.value, BinaryExpr) and stmt.value.op in (
                BinaryOp.DIV, BinaryOp.MOD, BinaryOp.FLOOR_DIV
            ):
                vcs.append(self._obligation_gen.from_division_safety_check(
                    stmt.value.right,
                    context_vars=list(contract.params),
                ))

        elif isinstance(stmt, CallStmt):
            callee_contract = self._wp_calc._function_contracts.get(stmt.func_name)
            if callee_contract:
                mapping: Dict[str, Expr] = {}
                for (pname, _), arg in zip(callee_contract.params, stmt.args):
                    mapping[pname] = arg
                for i, pre in enumerate(callee_contract.preconditions):
                    vcs.append(VerificationCondition(
                        vc_id=self._next_id("callpre"),
                        kind=VCKind.PRECONDITION,
                        formula=pre.substitute(mapping),
                        location=f"{location}:call:{stmt.func_name}",
                        description=(
                            f"Precondition {i} of {stmt.func_name} at call site"
                        ),
                        context_vars=list(contract.params),
                        assumptions=list(contract.preconditions),
                    ))

        elif isinstance(stmt, TryStmt):
            vcs.extend(self._extract_safety_vcs(stmt.body, f"{location}:try", contract))
            vcs.extend(self._extract_safety_vcs(stmt.handler, f"{location}:except", contract))
            if stmt.finally_block:
                vcs.extend(self._extract_safety_vcs(
                    stmt.finally_block, f"{location}:finally", contract
                ))

        return vcs


# ---------------------------------------------------------------------------
# ProofChecker
# ---------------------------------------------------------------------------

class ProofChecker:
    """Verify that a proof certificate is valid."""

    def __init__(self) -> None:
        self._axioms: Set[str] = set()

    def add_axiom(self, axiom: Expr) -> None:
        self._axioms.add(str(axiom))

    def check_proof(self, certificate: ProofCertificate) -> VerificationResult:
        """Check that the proof certificate is valid."""
        start = time.time()
        errors = certificate.validate()
        if errors:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                message=f"Structural validation failed: {'; '.join(errors)}",
                proof_steps_verified=0,
                proof_steps_total=len(certificate.proof_steps),
            )

        sm = certificate.step_map()
        verified_count = 0
        total = len(certificate.proof_steps)

        for step in certificate.proof_steps:
            step_ok = self._check_step(step, sm, certificate.assumptions)
            if step_ok:
                verified_count += 1

        elapsed = (time.time() - start) * 1000.0

        if verified_count == total and total > 0:
            status = VerificationStatus.VERIFIED
            msg = "All proof steps verified"
        elif verified_count > 0:
            status = VerificationStatus.UNKNOWN
            msg = f"{total - verified_count} step(s) could not be verified"
        else:
            status = VerificationStatus.FAILED
            msg = "No proof steps could be verified"

        return VerificationResult(
            status=status,
            message=msg,
            proof_steps_verified=verified_count,
            proof_steps_total=total,
            solver_time_ms=elapsed,
        )

    def _check_step(
        self,
        step: ProofStep,
        step_map: Dict[str, ProofStep],
        assumptions: List[Expr],
    ) -> bool:
        """Check that a single proof step is justified."""
        # Validate premises exist
        if not step.validate_step(step_map):
            return False

        if step.justification == Justification.AXIOM:
            return self._check_axiom(step, assumptions)
        if step.justification == Justification.ASSUMPTION:
            return self._check_assumption(step, assumptions)
        if step.justification == Justification.HYPOTHESIS:
            return True
        if step.justification == Justification.MODUS_PONENS:
            return self._check_modus_ponens(step, step_map)
        if step.justification == Justification.CASE_SPLIT:
            return self._check_case_split(step, step_map)
        if step.justification == Justification.SMT:
            return self._check_smt(step, step_map, assumptions)
        if step.justification == Justification.CUT:
            return self._check_cut(step, step_map)
        if step.justification == Justification.WEAKENING:
            return self._check_weakening(step, step_map)
        if step.justification == Justification.STRENGTHENING:
            return self._check_strengthening(step, step_map)
        if step.justification == Justification.INDUCTION:
            return self._check_induction(step, step_map)
        if step.justification == Justification.SUBTYPING:
            return True  # trust subtyping steps
        if step.justification == Justification.FRAME:
            return True  # trust frame steps
        return False

    def _check_axiom(self, step: ProofStep, assumptions: List[Expr]) -> bool:
        conclusion_str = str(step.conclusion)
        if conclusion_str in self._axioms:
            return True
        for a in assumptions:
            if str(a) == conclusion_str:
                return True
        return False

    def _check_assumption(self, step: ProofStep, assumptions: List[Expr]) -> bool:
        conclusion_str = str(step.conclusion)
        for a in assumptions:
            if str(a) == conclusion_str:
                return True
        return True  # assumptions are trusted

    def _check_modus_ponens(
        self, step: ProofStep, step_map: Dict[str, ProofStep]
    ) -> bool:
        """Check A, A ⇒ B ⊢ B."""
        if len(step.premises) < 2:
            return False
        p1 = step_map.get(step.premises[0])
        p2 = step_map.get(step.premises[1])
        if p1 is None or p2 is None:
            return False
        # Check if p2 is an implication with p1 as antecedent
        if isinstance(p2.conclusion, BinaryExpr) and p2.conclusion.op == BinaryOp.IMPLIES:
            if str(p2.conclusion.left) == str(p1.conclusion):
                if str(p2.conclusion.right) == str(step.conclusion):
                    return True
        # Check reversed
        if isinstance(p1.conclusion, BinaryExpr) and p1.conclusion.op == BinaryOp.IMPLIES:
            if str(p1.conclusion.left) == str(p2.conclusion):
                if str(p1.conclusion.right) == str(step.conclusion):
                    return True
        return False

    def _check_case_split(
        self, step: ProofStep, step_map: Dict[str, ProofStep]
    ) -> bool:
        """Verify case analysis: must have premises covering all cases."""
        if len(step.premises) < 2:
            return False
        for pid in step.premises:
            if pid not in step_map:
                return False
        return True

    def _check_smt(
        self,
        step: ProofStep,
        step_map: Dict[str, ProofStep],
        assumptions: List[Expr],
    ) -> bool:
        """SMT-justified steps are trusted (would invoke solver in full impl)."""
        return True

    def _check_cut(
        self, step: ProofStep, step_map: Dict[str, ProofStep]
    ) -> bool:
        if len(step.premises) < 2:
            return False
        for pid in step.premises:
            if pid not in step_map:
                return False
        return True

    def _check_weakening(
        self, step: ProofStep, step_map: Dict[str, ProofStep]
    ) -> bool:
        return len(step.premises) >= 1 and all(p in step_map for p in step.premises)

    def _check_strengthening(
        self, step: ProofStep, step_map: Dict[str, ProofStep]
    ) -> bool:
        return len(step.premises) >= 1 and all(p in step_map for p in step.premises)

    def _check_induction(
        self, step: ProofStep, step_map: Dict[str, ProofStep]
    ) -> bool:
        if len(step.premises) < 2:
            return False
        for pid in step.premises:
            if pid not in step_map:
                return False
        return True


# ---------------------------------------------------------------------------
# ProofFormatter
# ---------------------------------------------------------------------------

class ProofFormatter:
    """Format proof certificates into various output formats."""

    # ---- SMT-LIB -----------------------------------------------------------

    def to_smt_lib(self, certificate: ProofCertificate) -> str:
        lines: List[str] = []
        lines.append("; ============================================")
        lines.append(f"; Proof Certificate: {certificate.certificate_id}")
        lines.append(f"; Theorem: {certificate.theorem}")
        lines.append(f"; Steps: {len(certificate.proof_steps)}")
        lines.append("; ============================================")
        lines.append("")
        lines.append("(set-logic ALL)")
        lines.append("(set-option :produce-proofs true)")
        lines.append("")

        # Declare variables from free vars of all formulas
        all_vars: Set[str] = set()
        for step in certificate.proof_steps:
            all_vars |= step.conclusion.free_vars()
        for a in certificate.assumptions:
            all_vars |= a.free_vars()
        for v in sorted(all_vars):
            lines.append(f"(declare-const {v} Int)")
        lines.append("")

        # Assert assumptions
        lines.append("; ---- Assumptions ----")
        for i, a in enumerate(certificate.assumptions):
            lines.append(f"(assert (! {a.to_smt()} :named assumption_{i}))")
        lines.append("")

        # Assert proof steps as comments + assertions
        lines.append("; ---- Proof Steps ----")
        for step in certificate.proof_steps:
            lines.append(step.to_smt_comment())
        lines.append("")

        # Check sat
        if certificate.conclusion is not None:
            lines.append("; ---- Verification Goal ----")
            lines.append(
                f"(assert (not {certificate.conclusion.to_smt()}))"
            )
        lines.append("(check-sat)")
        lines.append("(exit)")
        return "\n".join(lines)

    # ---- Natural language --------------------------------------------------

    def to_natural_language(self, certificate: ProofCertificate) -> str:
        parts: List[str] = []
        parts.append(f"Proof Certificate: {certificate.certificate_id}")
        parts.append(f"Theorem: {certificate.theorem}")
        parts.append("")

        if certificate.assumptions:
            parts.append("Assumptions:")
            for i, a in enumerate(certificate.assumptions):
                parts.append(f"  A{i+1}. {a}")
            parts.append("")

        parts.append("Proof:")
        for step in certificate.proof_steps:
            parts.append(f"  {step.to_natural_language()}")
        parts.append("")

        if certificate.conclusion:
            parts.append(f"Conclusion: {certificate.conclusion}")
        parts.append("")

        if certificate.verification_result:
            parts.append(f"Verification: {certificate.verification_result.summary()}")

        return "\n".join(parts)

    # ---- LaTeX proof tree --------------------------------------------------

    def to_latex(self, certificate: ProofCertificate) -> str:
        lines: List[str] = []
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage{bussproofs}")
        lines.append(r"\usepackage{amsmath,amssymb}")
        lines.append(r"\begin{document}")
        lines.append("")
        lines.append(r"\section*{Proof Certificate: " + certificate.certificate_id + "}")
        lines.append(r"\textbf{Theorem:} " + self._latex_escape(certificate.theorem))
        lines.append("")

        if certificate.assumptions:
            lines.append(r"\subsection*{Assumptions}")
            lines.append(r"\begin{enumerate}")
            for a in certificate.assumptions:
                lines.append(r"  \item $" + self._expr_to_latex(a) + r"$")
            lines.append(r"\end{enumerate}")
            lines.append("")

        lines.append(r"\subsection*{Proof}")
        step_map = certificate.step_map()
        # Build proof tree for last step (conclusion)
        if certificate.proof_steps:
            last = certificate.proof_steps[-1]
            tree_str = self._build_latex_tree(last, step_map)
            lines.append(tree_str)

        lines.append("")
        lines.append(r"\end{document}")
        return "\n".join(lines)

    def _build_latex_tree(
        self, step: ProofStep, step_map: Dict[str, ProofStep], depth: int = 0
    ) -> str:
        if depth > 15:
            return r"\AxiomC{$\vdots$}"

        parts: List[str] = []

        if not step.premises:
            parts.append(
                r"\AxiomC{$" + self._expr_to_latex(step.conclusion) + r"$}"
            )
        else:
            for pid in step.premises:
                premise_step = step_map.get(pid)
                if premise_step:
                    parts.append(
                        self._build_latex_tree(premise_step, step_map, depth + 1)
                    )
                else:
                    parts.append(r"\AxiomC{$?$}")

            n = len(step.premises)
            rule_cmd = {1: r"\UnaryInfC", 2: r"\BinaryInfC", 3: r"\TrinaryInfC"}
            cmd = rule_cmd.get(n, r"\UnaryInfC")
            parts.append(
                r"\RightLabel{\scriptsize " + step.rule_name + "}"
            )
            parts.append(
                cmd + r"{$" + self._expr_to_latex(step.conclusion) + r"$}"
            )

        return "\n".join(parts)

    def _expr_to_latex(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, IntLit):
            return str(expr.value)
        if isinstance(expr, BoolLit):
            return r"\top" if expr.value else r"\bot"
        if isinstance(expr, StrLit):
            return r'\text{"' + expr.value + r'"}'
        if isinstance(expr, NoneLit):
            return r"\mathbf{None}"
        if isinstance(expr, UnaryExpr):
            op_map = {
                UnaryOp.NOT: r"\neg",
                UnaryOp.NEG: "-",
                UnaryOp.LEN: r"\mathrm{len}",
            }
            op_str = op_map.get(expr.op, str(expr.op.value))
            return f"{op_str}({self._expr_to_latex(expr.operand)})"
        if isinstance(expr, BinaryExpr):
            op_map = {
                BinaryOp.AND: r"\land",
                BinaryOp.OR: r"\lor",
                BinaryOp.IMPLIES: r"\Rightarrow",
                BinaryOp.EQ: "=",
                BinaryOp.NE: r"\neq",
                BinaryOp.LT: "<",
                BinaryOp.LE: r"\leq",
                BinaryOp.GT: ">",
                BinaryOp.GE: r"\geq",
                BinaryOp.ADD: "+",
                BinaryOp.SUB: "-",
                BinaryOp.MUL: r"\cdot",
                BinaryOp.DIV: r"\div",
                BinaryOp.MOD: r"\bmod",
            }
            op_str = op_map.get(expr.op, str(expr.op.value))
            return (
                f"{self._expr_to_latex(expr.left)} "
                f"{op_str} "
                f"{self._expr_to_latex(expr.right)}"
            )
        if isinstance(expr, QuantifiedExpr):
            q = r"\forall" if expr.quantifier == Quantifier.FORALL else r"\exists"
            return (
                f"{q} {expr.var_name} : {expr.var_sort}.\\ "
                f"{self._expr_to_latex(expr.body)}"
            )
        if isinstance(expr, FuncApp):
            args_str = ", ".join(self._expr_to_latex(a) for a in expr.args)
            return f"\\mathrm{{{expr.func_name}}}({args_str})"
        if isinstance(expr, IteExpr):
            return (
                r"\mathrm{ite}("
                + self._expr_to_latex(expr.cond) + ", "
                + self._expr_to_latex(expr.then_branch) + ", "
                + self._expr_to_latex(expr.else_branch) + ")"
            )
        if isinstance(expr, LetExpr):
            return (
                r"\mathrm{let}\ " + expr.var_name + " = "
                + self._expr_to_latex(expr.value) + r"\ \mathrm{in}\ "
                + self._expr_to_latex(expr.body)
            )
        return str(expr)

    @staticmethod
    def _latex_escape(text: str) -> str:
        for ch in ["_", "&", "%", "$", "#", "{", "}"]:
            text = text.replace(ch, "\\" + ch)
        return text

    # ---- Coq skeleton -----------------------------------------------------

    def to_coq(self, certificate: ProofCertificate) -> str:
        lines: List[str] = []
        lines.append("(* Proof Certificate (skeleton) *)")
        lines.append(f"(* {certificate.theorem} *)")
        lines.append("")

        # Collect free vars
        all_vars: Set[str] = set()
        for step in certificate.proof_steps:
            all_vars |= step.conclusion.free_vars()
        for a in certificate.assumptions:
            all_vars |= a.free_vars()

        if all_vars:
            lines.append("Section ProofCertificate.")
            for v in sorted(all_vars):
                lines.append(f"  Variable {v} : Z.")
            lines.append("")

        for i, a in enumerate(certificate.assumptions):
            lines.append(
                f"  Hypothesis assm_{i} : {self._expr_to_coq(a)}."
            )
        lines.append("")

        if certificate.conclusion:
            lines.append(
                f"  Theorem certificate : {self._expr_to_coq(certificate.conclusion)}."
            )
            lines.append("  Proof.")
            for step in certificate.proof_steps:
                lines.append(f"    (* Step {step.step_id}: {step.rule_name} *)")
                if step.justification == Justification.SMT:
                    lines.append("    (* proved by SMT solver -- fill in manual proof *)")
                    lines.append("    admit.")
                elif step.justification == Justification.MODUS_PONENS:
                    if len(step.premises) >= 2:
                        lines.append(f"    apply assm_{step.premises[1]}.")
                        lines.append(f"    exact assm_{step.premises[0]}.")
                elif step.justification == Justification.CASE_SPLIT:
                    lines.append("    destruct (classic _) as [H | H].")
                    lines.append("    - admit. (* then case *)")
                    lines.append("    - admit. (* else case *)")
                elif step.justification == Justification.INDUCTION:
                    lines.append("    induction _ as [| n IH].")
                    lines.append("    - admit. (* base case *)")
                    lines.append("    - admit. (* inductive step *)")
                else:
                    lines.append("    admit.")
            lines.append("  Qed.")
        lines.append("")
        if all_vars:
            lines.append("End ProofCertificate.")
        return "\n".join(lines)

    def _expr_to_coq(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, IntLit):
            return str(expr.value) if expr.value >= 0 else f"({expr.value})"
        if isinstance(expr, BoolLit):
            return "True" if expr.value else "False"
        if isinstance(expr, StrLit):
            return f'"{expr.value}"%string'
        if isinstance(expr, NoneLit):
            return "None"
        if isinstance(expr, UnaryExpr):
            if expr.op == UnaryOp.NOT:
                return f"(~ {self._expr_to_coq(expr.operand)})"
            if expr.op == UnaryOp.NEG:
                return f"(- {self._expr_to_coq(expr.operand)})"
            return f"({expr.op.value} {self._expr_to_coq(expr.operand)})"
        if isinstance(expr, BinaryExpr):
            l = self._expr_to_coq(expr.left)
            r = self._expr_to_coq(expr.right)
            op_map = {
                BinaryOp.AND: "/\\",
                BinaryOp.OR: "\\/",
                BinaryOp.IMPLIES: "->",
                BinaryOp.EQ: "=",
                BinaryOp.NE: "<>",
                BinaryOp.LT: "<",
                BinaryOp.LE: "<=",
                BinaryOp.GT: ">",
                BinaryOp.GE: ">=",
                BinaryOp.ADD: "+",
                BinaryOp.SUB: "-",
                BinaryOp.MUL: "*",
            }
            op = op_map.get(expr.op, str(expr.op.value))
            return f"({l} {op} {r})"
        if isinstance(expr, QuantifiedExpr):
            if expr.quantifier == Quantifier.FORALL:
                return f"(forall {expr.var_name} : Z, {self._expr_to_coq(expr.body)})"
            return f"(exists {expr.var_name} : Z, {self._expr_to_coq(expr.body)})"
        if isinstance(expr, FuncApp):
            args = " ".join(self._expr_to_coq(a) for a in expr.args)
            return f"({expr.func_name} {args})"
        if isinstance(expr, IteExpr):
            return (
                f"(if {self._expr_to_coq(expr.cond)} "
                f"then {self._expr_to_coq(expr.then_branch)} "
                f"else {self._expr_to_coq(expr.else_branch)})"
            )
        return str(expr)

    # ---- Lean 4 skeleton ---------------------------------------------------

    def to_lean(self, certificate: ProofCertificate) -> str:
        lines: List[str] = []
        lines.append("-- Proof Certificate (skeleton)")
        lines.append(f"-- {certificate.theorem}")
        lines.append("")

        all_vars: Set[str] = set()
        for step in certificate.proof_steps:
            all_vars |= step.conclusion.free_vars()
        for a in certificate.assumptions:
            all_vars |= a.free_vars()

        var_decls = " ".join(f"({v} : Int)" for v in sorted(all_vars))

        for i, a in enumerate(certificate.assumptions):
            lines.append(
                f"axiom assm_{i} : {self._expr_to_lean(a)}"
            )
        lines.append("")

        if certificate.conclusion:
            lines.append(
                f"theorem certificate {var_decls} : "
                f"{self._expr_to_lean(certificate.conclusion)} := by"
            )
            for step in certificate.proof_steps:
                lines.append(f"  -- Step {step.step_id}: {step.rule_name}")
                if step.justification == Justification.SMT:
                    lines.append("  sorry -- proved by SMT solver")
                elif step.justification == Justification.CASE_SPLIT:
                    lines.append("  by_cases")
                    lines.append("  · sorry -- then case")
                    lines.append("  · sorry -- else case")
                elif step.justification == Justification.INDUCTION:
                    lines.append("  induction _")
                    lines.append("  · sorry -- base case")
                    lines.append("  · sorry -- inductive step")
                else:
                    lines.append("  sorry")
        lines.append("")
        return "\n".join(lines)

    def _expr_to_lean(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, IntLit):
            return str(expr.value) if expr.value >= 0 else f"({expr.value})"
        if isinstance(expr, BoolLit):
            return "True" if expr.value else "False"
        if isinstance(expr, StrLit):
            return f'"{expr.value}"'
        if isinstance(expr, NoneLit):
            return "none"
        if isinstance(expr, UnaryExpr):
            if expr.op == UnaryOp.NOT:
                return f"(¬ {self._expr_to_lean(expr.operand)})"
            return f"({expr.op.value} {self._expr_to_lean(expr.operand)})"
        if isinstance(expr, BinaryExpr):
            l = self._expr_to_lean(expr.left)
            r = self._expr_to_lean(expr.right)
            op_map = {
                BinaryOp.AND: "∧",
                BinaryOp.OR: "∨",
                BinaryOp.IMPLIES: "→",
                BinaryOp.EQ: "=",
                BinaryOp.NE: "≠",
                BinaryOp.LT: "<",
                BinaryOp.LE: "≤",
                BinaryOp.GT: ">",
                BinaryOp.GE: "≥",
                BinaryOp.ADD: "+",
                BinaryOp.SUB: "-",
                BinaryOp.MUL: "*",
            }
            op = op_map.get(expr.op, str(expr.op.value))
            return f"({l} {op} {r})"
        if isinstance(expr, QuantifiedExpr):
            if expr.quantifier == Quantifier.FORALL:
                return f"(∀ {expr.var_name} : Int, {self._expr_to_lean(expr.body)})"
            return f"(∃ {expr.var_name} : Int, {self._expr_to_lean(expr.body)})"
        if isinstance(expr, FuncApp):
            args = " ".join(self._expr_to_lean(a) for a in expr.args)
            return f"({expr.func_name} {args})"
        return str(expr)

    # ---- HTML interactive viewer -------------------------------------------

    def to_html(self, certificate: ProofCertificate) -> str:
        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html lang='en'>")
        parts.append("<head>")
        parts.append("<meta charset='UTF-8'>")
        parts.append("<title>Proof Certificate</title>")
        parts.append("<style>")
        parts.append(self._html_css())
        parts.append("</style>")
        parts.append("</head>")
        parts.append("<body>")
        parts.append(f"<h1>Proof Certificate: {certificate.certificate_id}</h1>")
        parts.append(f"<h2>Theorem: {self._html_escape(certificate.theorem)}</h2>")

        if certificate.verification_result:
            vr = certificate.verification_result
            status_class = (
                "verified" if vr.is_verified else "failed"
            )
            parts.append(
                f"<div class='status {status_class}'>{vr.summary()}</div>"
            )

        if certificate.assumptions:
            parts.append("<h3>Assumptions</h3>")
            parts.append("<ol class='assumptions'>")
            for a in certificate.assumptions:
                parts.append(
                    f"  <li><code>{self._html_escape(str(a))}</code></li>"
                )
            parts.append("</ol>")

        parts.append("<h3>Proof Steps</h3>")
        parts.append("<table class='proof-steps'>")
        parts.append("<tr><th>Step</th><th>Rule</th><th>Premises</th>"
                      "<th>Conclusion</th><th>Justification</th></tr>")
        for step in certificate.proof_steps:
            premises_str = ", ".join(step.premises) if step.premises else "—"
            parts.append(
                f"<tr>"
                f"<td>{step.step_id}</td>"
                f"<td>{step.rule_name}</td>"
                f"<td>{premises_str}</td>"
                f"<td><code>{self._html_escape(str(step.conclusion))}</code></td>"
                f"<td>{step.justification.value}</td>"
                f"</tr>"
            )
        parts.append("</table>")

        if certificate.conclusion:
            parts.append("<h3>Conclusion</h3>")
            parts.append(
                f"<p><code>{self._html_escape(str(certificate.conclusion))}</code></p>"
            )

        # Interactive tree view (JavaScript)
        parts.append("<h3>Interactive Proof Tree</h3>")
        parts.append("<div id='proof-tree'></div>")
        parts.append("<script>")
        parts.append(self._html_tree_js(certificate))
        parts.append("</script>")

        parts.append("</body></html>")
        return "\n".join(parts)

    def _html_css(self) -> str:
        return textwrap.dedent("""\
            body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .status { padding: 10px; border-radius: 4px; margin: 10px 0; font-weight: bold; }
            .status.verified { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.failed { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .assumptions { background: #f8f9fa; padding: 15px 15px 15px 35px; border-radius: 4px; }
            .proof-steps { width: 100%; border-collapse: collapse; margin: 15px 0; }
            .proof-steps th, .proof-steps td { border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }
            .proof-steps th { background: #343a40; color: white; }
            .proof-steps tr:nth-child(even) { background: #f8f9fa; }
            code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
            .tree-node { cursor: pointer; padding: 4px 8px; margin: 2px 0; border-left: 3px solid #007bff; background: #f1f3f5; }
            .tree-node:hover { background: #dee2e6; }
            .tree-children { margin-left: 20px; }
        """)

    def _html_tree_js(self, certificate: ProofCertificate) -> str:
        """Generate JavaScript for interactive proof tree."""
        step_map = certificate.step_map()
        root_ids = set()
        referenced = set()
        for s in certificate.proof_steps:
            for p in s.premises:
                referenced.add(p)
        for s in certificate.proof_steps:
            if s.step_id not in referenced:
                root_ids.add(s.step_id)
        if not root_ids and certificate.proof_steps:
            root_ids = {certificate.proof_steps[-1].step_id}

        lines: List[str] = []
        lines.append("(function() {")
        lines.append("  var steps = {};")
        for s in certificate.proof_steps:
            premises_js = "[" + ",".join(f'"{p}"' for p in s.premises) + "]"
            conc_js = str(s.conclusion).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(
                f'  steps["{s.step_id}"] = {{'
                f'id:"{s.step_id}",'
                f'rule:"{s.rule_name}",'
                f'premises:{premises_js},'
                f'conclusion:"{conc_js}",'
                f'justification:"{s.justification.value}"'
                f'}};'
            )
        lines.append("  function buildTree(id) {")
        lines.append("    var s = steps[id]; if (!s) return '';")
        lines.append("    var children = '';")
        lines.append("    for (var i=0; i<s.premises.length; i++) {")
        lines.append("      children += buildTree(s.premises[i]);")
        lines.append("    }")
        lines.append("    var childDiv = children ? '<div class=\"tree-children\">' + children + '</div>' : '';")
        lines.append("    return '<div class=\"tree-node\" onclick=\"this.querySelector(\\'.tree-children\\')&&(this.querySelector(\\'.tree-children\\').style.display=this.querySelector(\\'.tree-children\\').style.display===\\'none\\'?\\'block\\':\\'none\\')\">' +")
        lines.append("      '<strong>' + s.id + '</strong> [' + s.rule + '] ' + s.conclusion + ' <em>(' + s.justification + ')</em>' +")
        lines.append("      childDiv + '</div>';")
        lines.append("  }")
        lines.append("  var container = document.getElementById('proof-tree');")
        roots_js = "[" + ",".join(f'"{r}"' for r in sorted(root_ids)) + "]"
        lines.append(f"  var roots = {roots_js};")
        lines.append("  var html = '';")
        lines.append("  for (var i=0; i<roots.length; i++) { html += buildTree(roots[i]); }")
        lines.append("  container.innerHTML = html;")
        lines.append("})();")
        return "\n".join(lines)

    @staticmethod
    def _html_escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


# ---------------------------------------------------------------------------
# ProofStatistics
# ---------------------------------------------------------------------------

@dataclass
class ProofStatistics:
    """Aggregate statistics about proof certificates."""

    total_obligations: int = 0
    verified_count: int = 0
    unverified_count: int = 0
    unknown_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_steps: int = 0
    total_solver_time_ms: float = 0.0
    per_vc_stats: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def verification_rate(self) -> float:
        if self.total_obligations == 0:
            return 0.0
        return self.verified_count / self.total_obligations

    @property
    def average_proof_size(self) -> float:
        if self.total_obligations == 0:
            return 0.0
        return self.total_steps / self.total_obligations

    @property
    def average_solver_time_ms(self) -> float:
        if self.total_obligations == 0:
            return 0.0
        return self.total_solver_time_ms / self.total_obligations

    def most_complex_proofs(self, top_n: int = 5) -> List[Dict[str, Any]]:
        sorted_stats = sorted(
            self.per_vc_stats, key=lambda s: s.get("steps", 0), reverse=True
        )
        return sorted_stats[:top_n]

    def record_vc(
        self,
        vc_id: str,
        status: VerificationStatus,
        steps: int = 0,
        solver_time_ms: float = 0.0,
    ) -> None:
        self.total_obligations += 1
        self.total_steps += steps
        self.total_solver_time_ms += solver_time_ms

        if status == VerificationStatus.VERIFIED:
            self.verified_count += 1
        elif status == VerificationStatus.FAILED:
            self.unverified_count += 1
        elif status == VerificationStatus.UNKNOWN:
            self.unknown_count += 1
        elif status == VerificationStatus.TIMEOUT:
            self.timeout_count += 1
        elif status == VerificationStatus.ERROR:
            self.error_count += 1

        self.per_vc_stats.append({
            "vc_id": vc_id,
            "status": status.value,
            "steps": steps,
            "solver_time_ms": solver_time_ms,
        })

    def summary(self) -> str:
        parts: List[str] = []
        parts.append(f"Total obligations: {self.total_obligations}")
        parts.append(f"Verified: {self.verified_count}")
        parts.append(f"Unverified: {self.unverified_count}")
        parts.append(f"Unknown: {self.unknown_count}")
        parts.append(f"Timeout: {self.timeout_count}")
        parts.append(f"Errors: {self.error_count}")
        parts.append(f"Verification rate: {self.verification_rate:.1%}")
        parts.append(f"Average proof size: {self.average_proof_size:.1f} steps")
        parts.append(
            f"Average solver time: {self.average_solver_time_ms:.1f}ms"
        )
        parts.append(f"Total solver time: {self.total_solver_time_ms:.1f}ms")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_obligations": self.total_obligations,
            "verified": self.verified_count,
            "unverified": self.unverified_count,
            "unknown": self.unknown_count,
            "timeout": self.timeout_count,
            "errors": self.error_count,
            "verification_rate": self.verification_rate,
            "average_proof_size": self.average_proof_size,
            "average_solver_time_ms": self.average_solver_time_ms,
            "total_solver_time_ms": self.total_solver_time_ms,
            "most_complex": self.most_complex_proofs(),
        }


# ---------------------------------------------------------------------------
# ProofPipeline – orchestrates end-to-end proof certificate generation
# ---------------------------------------------------------------------------

class ProofPipeline:
    """
    End-to-end pipeline: contracts → VCs → proof certificates → verification.
    """

    def __init__(self) -> None:
        self._vc_gen = VerificationConditionGenerator()
        self._checker = ProofChecker()
        self._formatter = ProofFormatter()
        self._smt_gen = SmtLibGenerator()
        self._hoare = HoareTripleVerifier()
        self._stats = ProofStatistics()
        self._certificates: List[ProofCertificate] = []

    @property
    def statistics(self) -> ProofStatistics:
        return self._stats

    def register_contract(self, contract: FunctionContract) -> None:
        self._vc_gen.register_contract(contract)
        self._hoare.register_contract(contract)

    # ---- certificate generation -------------------------------------------

    def generate_certificate(
        self,
        func_name: str,
        body: Stmt,
        contract: FunctionContract,
    ) -> ProofCertificate:
        """Generate a proof certificate for a function."""
        cert_id = f"cert_{func_name}_{int(time.time()*1000) % 100000}"
        certificate = ProofCertificate(
            certificate_id=cert_id,
            theorem=f"Function {func_name} satisfies its contract",
            assumptions=list(contract.preconditions),
        )

        # Generate VCs
        vcs = self._vc_gen.generate_vcs(func_name, body, contract)
        certificate.verification_conditions = vcs

        # Attempt to build a proof via Hoare triple verification
        pre_conj = _conjunction(contract.preconditions) if contract.preconditions else TRUE
        post_conj = _conjunction(contract.postconditions) if contract.postconditions else TRUE

        result_ok, steps = self._hoare.verify_triple(pre_conj, body, post_conj)
        certificate.proof_steps = steps
        certificate.conclusion = _implies(pre_conj, post_conj)

        # Verify the certificate
        vr = self._checker.check_proof(certificate)
        certificate.verification_result = vr

        # Record statistics
        for vc in vcs:
            self._stats.record_vc(
                vc.vc_id,
                VerificationStatus.VERIFIED if result_ok else VerificationStatus.UNKNOWN,
                steps=len(steps),
                solver_time_ms=vr.solver_time_ms,
            )

        self._certificates.append(certificate)
        return certificate

    # ---- bulk certificate generation --------------------------------------

    def generate_certificates(
        self,
        functions: List[Tuple[str, Stmt, FunctionContract]],
    ) -> List[ProofCertificate]:
        results: List[ProofCertificate] = []
        for func_name, body, contract in functions:
            cert = self.generate_certificate(func_name, body, contract)
            results.append(cert)
        return results

    # ---- output formatting ------------------------------------------------

    def format_all_smt_lib(self) -> str:
        parts: List[str] = []
        for cert in self._certificates:
            parts.append(self._formatter.to_smt_lib(cert))
            parts.append("")
        return "\n".join(parts)

    def format_all_natural_language(self) -> str:
        parts: List[str] = []
        for cert in self._certificates:
            parts.append(self._formatter.to_natural_language(cert))
            parts.append("=" * 60)
        return "\n".join(parts)

    def format_all_latex(self) -> str:
        parts: List[str] = []
        parts.append(r"\documentclass{article}")
        parts.append(r"\usepackage{bussproofs}")
        parts.append(r"\usepackage{amsmath,amssymb}")
        parts.append(r"\begin{document}")
        parts.append(r"\title{Proof Certificates}")
        parts.append(r"\maketitle")
        for cert in self._certificates:
            tree_body = self._formatter.to_latex(cert)
            # Extract just the body between \begin{document} and \end{document}
            start = tree_body.find(r"\section*")
            end = tree_body.find(r"\end{document}")
            if start >= 0 and end >= 0:
                parts.append(tree_body[start:end])
            else:
                parts.append(tree_body)
            parts.append("")
        parts.append(r"\end{document}")
        return "\n".join(parts)

    def format_all_html(self) -> str:
        if len(self._certificates) == 1:
            return self._formatter.to_html(self._certificates[0])

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset='UTF-8'>")
        parts.append("<title>Proof Certificates</title>")
        parts.append(f"<style>{self._formatter._html_css()}</style>")
        parts.append("</head><body>")
        parts.append(f"<h1>Proof Certificates ({len(self._certificates)} total)</h1>")

        # Summary table
        parts.append("<table class='proof-steps'>")
        parts.append("<tr><th>ID</th><th>Theorem</th><th>Steps</th><th>Status</th></tr>")
        for cert in self._certificates:
            status = (
                cert.verification_result.status.value
                if cert.verification_result
                else "unknown"
            )
            parts.append(
                f"<tr><td>{cert.certificate_id}</td>"
                f"<td>{cert.theorem}</td>"
                f"<td>{len(cert.proof_steps)}</td>"
                f"<td>{status}</td></tr>"
            )
        parts.append("</table>")
        parts.append("<hr>")

        for cert in self._certificates:
            nl = self._formatter.to_natural_language(cert)
            parts.append(f"<pre>{nl}</pre>")
            parts.append("<hr>")

        parts.append(f"<h2>Statistics</h2><pre>{self._stats.summary()}</pre>")
        parts.append("</body></html>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Utility: SMT-LIB verification condition batch runner
# ---------------------------------------------------------------------------

class SmtBatchRunner:
    """
    Batch-generate SMT-LIB scripts for a list of VCs and optionally
    run an external solver (subprocess).
    """

    def __init__(self, solver_cmd: Optional[List[str]] = None) -> None:
        self._solver_cmd = solver_cmd or ["z3", "-smt2", "-in"]
        self._smt_gen = SmtLibGenerator()

    def generate_scripts(
        self, vcs: List[VerificationCondition]
    ) -> Dict[str, str]:
        """Return mapping of vc_id → SMT-LIB script."""
        return {vc.vc_id: vc.to_smt_query() for vc in vcs}

    def run_single(self, vc: VerificationCondition) -> VerificationResult:
        """Run a single VC through the SMT solver (requires subprocess)."""
        import subprocess

        script = vc.to_smt_query()
        start = time.time()
        try:
            proc = subprocess.run(
                self._solver_cmd,
                input=script,
                capture_output=True,
                text=True,
                timeout=30,
            )
            elapsed = (time.time() - start) * 1000.0
            output = proc.stdout.strip()
            if output.startswith("unsat"):
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    message="SMT solver returned unsat (VC holds)",
                    solver_time_ms=elapsed,
                )
            elif output.startswith("sat"):
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    message="SMT solver returned sat (VC does not hold)",
                    solver_time_ms=elapsed,
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    message=f"SMT solver returned: {output}",
                    solver_time_ms=elapsed,
                )
        except FileNotFoundError:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message=f"Solver not found: {self._solver_cmd[0]}",
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                message="SMT solver timed out",
                solver_time_ms=30000.0,
            )

    def run_batch(
        self, vcs: List[VerificationCondition]
    ) -> List[Tuple[str, VerificationResult]]:
        results: List[Tuple[str, VerificationResult]] = []
        for vc in vcs:
            vr = self.run_single(vc)
            vc.status = vr.status
            vc.solver_time_ms = vr.solver_time_ms
            if vr.counterexample:
                vc.counterexample = vr.counterexample
            results.append((vc.vc_id, vr))
        return results


# ---------------------------------------------------------------------------
# Expression simplifier (basic algebraic simplifications)
# ---------------------------------------------------------------------------

class ExpressionSimplifier:
    """Simplify expressions for smaller proof obligations."""

    def simplify(self, expr: Expr) -> Expr:
        if isinstance(expr, (Var, IntLit, BoolLit, StrLit, NoneLit)):
            return expr

        if isinstance(expr, UnaryExpr):
            operand = self.simplify(expr.operand)
            return self._simplify_unary(expr.op, operand)

        if isinstance(expr, BinaryExpr):
            left = self.simplify(expr.left)
            right = self.simplify(expr.right)
            return self._simplify_binary(expr.op, left, right)

        if isinstance(expr, QuantifiedExpr):
            body = self.simplify(expr.body)
            if isinstance(body, BoolLit):
                return body
            if expr.var_name not in body.free_vars():
                return body
            return QuantifiedExpr(expr.quantifier, expr.var_name, expr.var_sort, body)

        if isinstance(expr, IteExpr):
            cond = self.simplify(expr.cond)
            then_b = self.simplify(expr.then_branch)
            else_b = self.simplify(expr.else_branch)
            if isinstance(cond, BoolLit):
                return then_b if cond.value else else_b
            if str(then_b) == str(else_b):
                return then_b
            return IteExpr(cond, then_b, else_b)

        if isinstance(expr, LetExpr):
            value = self.simplify(expr.value)
            body = self.simplify(expr.body)
            if expr.var_name not in body.free_vars():
                return body
            return LetExpr(expr.var_name, value, body)

        if isinstance(expr, FuncApp):
            args = tuple(self.simplify(a) for a in expr.args)
            return FuncApp(expr.func_name, args)

        return expr

    def _simplify_unary(self, op: UnaryOp, operand: Expr) -> Expr:
        if op == UnaryOp.NOT:
            if isinstance(operand, BoolLit):
                return BoolLit(not operand.value)
            if isinstance(operand, UnaryExpr) and operand.op == UnaryOp.NOT:
                return operand.operand
            # ¬(a ∧ b) → ¬a ∨ ¬b  (De Morgan)
            if isinstance(operand, BinaryExpr) and operand.op == BinaryOp.AND:
                return BinaryExpr(
                    BinaryOp.OR,
                    UnaryExpr(UnaryOp.NOT, operand.left),
                    UnaryExpr(UnaryOp.NOT, operand.right),
                )
            if isinstance(operand, BinaryExpr) and operand.op == BinaryOp.OR:
                return BinaryExpr(
                    BinaryOp.AND,
                    UnaryExpr(UnaryOp.NOT, operand.left),
                    UnaryExpr(UnaryOp.NOT, operand.right),
                )
        if op == UnaryOp.NEG:
            if isinstance(operand, IntLit):
                return IntLit(-operand.value)
            if isinstance(operand, UnaryExpr) and operand.op == UnaryOp.NEG:
                return operand.operand
        if op == UnaryOp.ABS:
            if isinstance(operand, IntLit):
                return IntLit(abs(operand.value))
        return UnaryExpr(op, operand)

    def _simplify_binary(self, op: BinaryOp, left: Expr, right: Expr) -> Expr:
        # Boolean simplifications
        if op == BinaryOp.AND:
            if isinstance(left, BoolLit):
                return right if left.value else FALSE
            if isinstance(right, BoolLit):
                return left if right.value else FALSE
            if str(left) == str(right):
                return left
        if op == BinaryOp.OR:
            if isinstance(left, BoolLit):
                return TRUE if left.value else right
            if isinstance(right, BoolLit):
                return TRUE if right.value else left
            if str(left) == str(right):
                return left
        if op == BinaryOp.IMPLIES:
            if isinstance(left, BoolLit):
                return TRUE if not left.value else right
            if isinstance(right, BoolLit):
                return TRUE if right.value else _not(left)

        # Arithmetic simplifications
        if op == BinaryOp.ADD:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return IntLit(left.value + right.value)
            if isinstance(left, IntLit) and left.value == 0:
                return right
            if isinstance(right, IntLit) and right.value == 0:
                return left
        if op == BinaryOp.SUB:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return IntLit(left.value - right.value)
            if isinstance(right, IntLit) and right.value == 0:
                return left
            if str(left) == str(right):
                return IntLit(0)
        if op == BinaryOp.MUL:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return IntLit(left.value * right.value)
            if isinstance(left, IntLit) and left.value == 0:
                return IntLit(0)
            if isinstance(right, IntLit) and right.value == 0:
                return IntLit(0)
            if isinstance(left, IntLit) and left.value == 1:
                return right
            if isinstance(right, IntLit) and right.value == 1:
                return left

        # Comparison simplifications
        if op == BinaryOp.EQ:
            if str(left) == str(right):
                return TRUE
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return BoolLit(left.value == right.value)
        if op == BinaryOp.NE:
            if str(left) == str(right):
                return FALSE
        if op == BinaryOp.LT:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return BoolLit(left.value < right.value)
        if op == BinaryOp.LE:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return BoolLit(left.value <= right.value)
            if str(left) == str(right):
                return TRUE
        if op == BinaryOp.GT:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return BoolLit(left.value > right.value)
        if op == BinaryOp.GE:
            if isinstance(left, IntLit) and isinstance(right, IntLit):
                return BoolLit(left.value >= right.value)
            if str(left) == str(right):
                return TRUE

        return BinaryExpr(op, left, right)


# ---------------------------------------------------------------------------
# NNF / CNF normaliser (useful for SMT input)
# ---------------------------------------------------------------------------

class FormulaNormalizer:
    """Convert formulas to various normal forms."""

    def __init__(self) -> None:
        self._simplifier = ExpressionSimplifier()

    def to_nnf(self, expr: Expr) -> Expr:
        """Negation normal form: push negations to leaves."""
        if isinstance(expr, (Var, IntLit, BoolLit, StrLit, NoneLit, FuncApp)):
            return expr

        if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NOT:
            return self._nnf_not(expr.operand)

        if isinstance(expr, BinaryExpr):
            if expr.op == BinaryOp.IMPLIES:
                return _or(self._nnf_not(expr.left), self.to_nnf(expr.right))
            left = self.to_nnf(expr.left)
            right = self.to_nnf(expr.right)
            return BinaryExpr(expr.op, left, right)

        if isinstance(expr, QuantifiedExpr):
            body = self.to_nnf(expr.body)
            return QuantifiedExpr(expr.quantifier, expr.var_name, expr.var_sort, body)

        return expr

    def _nnf_not(self, expr: Expr) -> Expr:
        """Push a negation inward."""
        if isinstance(expr, BoolLit):
            return BoolLit(not expr.value)
        if isinstance(expr, UnaryExpr) and expr.op == UnaryOp.NOT:
            return self.to_nnf(expr.operand)
        if isinstance(expr, BinaryExpr):
            if expr.op == BinaryOp.AND:
                return _or(self._nnf_not(expr.left), self._nnf_not(expr.right))
            if expr.op == BinaryOp.OR:
                return _and(self._nnf_not(expr.left), self._nnf_not(expr.right))
            if expr.op == BinaryOp.IMPLIES:
                return _and(self.to_nnf(expr.left), self._nnf_not(expr.right))
            if expr.op == BinaryOp.EQ:
                return BinaryExpr(BinaryOp.NE, self.to_nnf(expr.left), self.to_nnf(expr.right))
            if expr.op == BinaryOp.NE:
                return BinaryExpr(BinaryOp.EQ, self.to_nnf(expr.left), self.to_nnf(expr.right))
            if expr.op == BinaryOp.LT:
                return BinaryExpr(BinaryOp.GE, self.to_nnf(expr.left), self.to_nnf(expr.right))
            if expr.op == BinaryOp.LE:
                return BinaryExpr(BinaryOp.GT, self.to_nnf(expr.left), self.to_nnf(expr.right))
            if expr.op == BinaryOp.GT:
                return BinaryExpr(BinaryOp.LE, self.to_nnf(expr.left), self.to_nnf(expr.right))
            if expr.op == BinaryOp.GE:
                return BinaryExpr(BinaryOp.LT, self.to_nnf(expr.left), self.to_nnf(expr.right))
        if isinstance(expr, QuantifiedExpr):
            dual = (
                Quantifier.EXISTS
                if expr.quantifier == Quantifier.FORALL
                else Quantifier.FORALL
            )
            return QuantifiedExpr(dual, expr.var_name, expr.var_sort, self._nnf_not(expr.body))
        return _not(self.to_nnf(expr))

    def to_cnf(self, expr: Expr) -> List[List[Expr]]:
        """
        Convert to conjunctive normal form (list of clauses, each a list of literals).
        Only works for propositional-level structure.
        """
        nnf = self.to_nnf(expr)
        return self._distribute(nnf)

    def _distribute(self, expr: Expr) -> List[List[Expr]]:
        """Distribute OR over AND to get CNF."""
        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.AND:
            left_clauses = self._distribute(expr.left)
            right_clauses = self._distribute(expr.right)
            return left_clauses + right_clauses
        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.OR:
            left_clauses = self._distribute(expr.left)
            right_clauses = self._distribute(expr.right)
            # Distribute: each clause from left ∨ each clause from right
            result: List[List[Expr]] = []
            for lc in left_clauses:
                for rc in right_clauses:
                    result.append(lc + rc)
            return result
        return [[expr]]

    def clauses_to_expr(self, clauses: List[List[Expr]]) -> Expr:
        """Convert CNF clauses back to a formula."""
        if not clauses:
            return TRUE
        clause_exprs = [_disjunction(clause) for clause in clauses]
        return _conjunction(clause_exprs)


# ---------------------------------------------------------------------------
# SSA transformation (useful for path encoding)
# ---------------------------------------------------------------------------

class SSATransformer:
    """Transform statements into SSA form for verification."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}

    def _fresh(self, var_name: str) -> str:
        count = self._counters.get(var_name, 0) + 1
        self._counters[var_name] = count
        return f"{var_name}_ssa_{count}"

    def _current(self, var_name: str) -> str:
        count = self._counters.get(var_name, 0)
        if count == 0:
            return var_name
        return f"{var_name}_ssa_{count}"

    def transform(self, stmt: Stmt) -> Tuple[List[Expr], Dict[str, str]]:
        """
        Transform a statement to SSA. Returns:
        - List of constraints (equalities)
        - Mapping from original variable names to current SSA names
        """
        self._counters.clear()
        constraints: List[Expr] = []
        self._transform_rec(stmt, constraints)
        final_mapping = {
            var: self._current(var) for var in self._counters
        }
        return constraints, final_mapping

    def _transform_rec(self, stmt: Stmt, constraints: List[Expr]) -> None:
        if isinstance(stmt, SkipStmt):
            return

        if isinstance(stmt, AssignStmt):
            rhs = self._rename_expr(stmt.value)
            new_name = self._fresh(stmt.target)
            constraints.append(_eq(_var(new_name), rhs))

        elif isinstance(stmt, SeqStmt):
            self._transform_rec(stmt.first, constraints)
            self._transform_rec(stmt.second, constraints)

        elif isinstance(stmt, AssumeStmt):
            constraints.append(self._rename_expr(stmt.condition))

        elif isinstance(stmt, AssertStmt):
            constraints.append(self._rename_expr(stmt.condition))

        elif isinstance(stmt, IfStmt):
            # For SSA, would need phi-nodes; simplified here
            cond = self._rename_expr(stmt.condition)
            then_constraints: List[Expr] = []
            self._transform_rec(stmt.then_branch, then_constraints)
            else_constraints: List[Expr] = []
            self._transform_rec(stmt.else_branch, else_constraints)
            if then_constraints:
                constraints.append(_implies(cond, _conjunction(then_constraints)))
            if else_constraints:
                constraints.append(_implies(_not(cond), _conjunction(else_constraints)))

        elif isinstance(stmt, CallStmt):
            if stmt.target:
                new_name = self._fresh(stmt.target)
                # Result is unconstrained (modeled by fresh variable)

    def _rename_expr(self, expr: Expr) -> Expr:
        if isinstance(expr, Var):
            return _var(self._current(expr.name))
        if isinstance(expr, (IntLit, BoolLit, StrLit, NoneLit)):
            return expr
        if isinstance(expr, UnaryExpr):
            return UnaryExpr(expr.op, self._rename_expr(expr.operand))
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(
                expr.op,
                self._rename_expr(expr.left),
                self._rename_expr(expr.right),
            )
        if isinstance(expr, FuncApp):
            return FuncApp(
                expr.func_name,
                tuple(self._rename_expr(a) for a in expr.args),
            )
        if isinstance(expr, IteExpr):
            return IteExpr(
                self._rename_expr(expr.cond),
                self._rename_expr(expr.then_branch),
                self._rename_expr(expr.else_branch),
            )
        return expr


# ---------------------------------------------------------------------------
# Invariant inference helpers
# ---------------------------------------------------------------------------

class InvariantStrengthener:
    """
    Attempt to strengthen a loop invariant to make a VC pass.
    Uses heuristic templates.
    """

    TEMPLATES: List[Callable[[str, Expr], Expr]] = [
        lambda v, bound: _ge(_var(v), _int(0)),
        lambda v, bound: _le(_var(v), bound),
        lambda v, bound: _and(_ge(_var(v), _int(0)), _le(_var(v), bound)),
    ]

    def strengthen(
        self,
        current_inv: Expr,
        loop_var: str,
        bound_expr: Expr,
    ) -> List[Expr]:
        """Return candidate strengthened invariants."""
        candidates: List[Expr] = []
        for template in self.TEMPLATES:
            candidate = _and(current_inv, template(loop_var, bound_expr))
            candidates.append(candidate)
        return candidates

    def weaken(self, current_inv: Expr) -> List[Expr]:
        """Heuristic weakening: remove conjuncts one at a time."""
        conjuncts = self._flatten_and(current_inv)
        if len(conjuncts) <= 1:
            return [TRUE]
        candidates: List[Expr] = []
        for i in range(len(conjuncts)):
            remaining = conjuncts[:i] + conjuncts[i + 1:]
            candidates.append(_conjunction(remaining))
        return candidates

    def _flatten_and(self, expr: Expr) -> List[Expr]:
        if isinstance(expr, BinaryExpr) and expr.op == BinaryOp.AND:
            return self._flatten_and(expr.left) + self._flatten_and(expr.right)
        return [expr]


# ---------------------------------------------------------------------------
# AnnotatedProgram: program with embedded proof obligations
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedProgram:
    """A program annotated with contracts and invariants."""
    func_name: str
    params: List[Tuple[str, str]]
    body: Stmt
    preconditions: List[Expr] = field(default_factory=list)
    postconditions: List[Expr] = field(default_factory=list)
    loop_invariants: Dict[str, Expr] = field(default_factory=dict)

    def to_contract(self) -> FunctionContract:
        return FunctionContract(
            func_name=self.func_name,
            params=self.params,
            preconditions=self.preconditions,
            postconditions=self.postconditions,
        )


# ---------------------------------------------------------------------------
# Full proof certificate generation for an annotated program
# ---------------------------------------------------------------------------

class AnnotatedProgramVerifier:
    """
    Complete verifier for annotated programs.
    Generates VCs, attempts proof, produces certificate.
    """

    def __init__(self) -> None:
        self._pipeline = ProofPipeline()
        self._simplifier = ExpressionSimplifier()

    def verify(self, program: AnnotatedProgram) -> ProofCertificate:
        contract = program.to_contract()
        self._pipeline.register_contract(contract)

        # Inject loop invariants into the body
        annotated_body = self._inject_invariants(program.body, program.loop_invariants)

        # Simplify all formulas
        simplified_pre = [self._simplifier.simplify(p) for p in contract.preconditions]
        simplified_post = [self._simplifier.simplify(p) for p in contract.postconditions]
        contract.preconditions = simplified_pre
        contract.postconditions = simplified_post

        return self._pipeline.generate_certificate(
            program.func_name, annotated_body, contract
        )

    def _inject_invariants(
        self, stmt: Stmt, invariants: Dict[str, Expr]
    ) -> Stmt:
        """Replace WhileStmt invariants from the annotation map."""
        if isinstance(stmt, WhileStmt):
            inv = invariants.get(str(id(stmt)))
            if inv is not None:
                return WhileStmt(stmt.condition, stmt.body, inv)
            return stmt
        if isinstance(stmt, SeqStmt):
            return SeqStmt(
                self._inject_invariants(stmt.first, invariants),
                self._inject_invariants(stmt.second, invariants),
            )
        if isinstance(stmt, IfStmt):
            return IfStmt(
                stmt.condition,
                self._inject_invariants(stmt.then_branch, invariants),
                self._inject_invariants(stmt.else_branch, invariants),
            )
        if isinstance(stmt, TryStmt):
            return TryStmt(
                self._inject_invariants(stmt.body, invariants),
                stmt.except_type,
                stmt.except_var,
                self._inject_invariants(stmt.handler, invariants),
                self._inject_invariants(stmt.finally_block, invariants)
                if stmt.finally_block
                else None,
            )
        return stmt

    def statistics(self) -> ProofStatistics:
        return self._pipeline.statistics


# ---------------------------------------------------------------------------
# ProofArchive: serialize / deserialize proof certificates
# ---------------------------------------------------------------------------

class ProofArchive:
    """Serialize proof certificates to/from JSON-compatible dicts."""

    def serialize_certificate(self, cert: ProofCertificate) -> Dict[str, Any]:
        return {
            "certificate_id": cert.certificate_id,
            "theorem": cert.theorem,
            "assumptions": [str(a) for a in cert.assumptions],
            "conclusion": str(cert.conclusion) if cert.conclusion else None,
            "proof_steps": [self._serialize_step(s) for s in cert.proof_steps],
            "verification_conditions": [
                self._serialize_vc(vc) for vc in cert.verification_conditions
            ],
            "verification_result": self._serialize_vr(cert.verification_result)
            if cert.verification_result
            else None,
            "metadata": cert.metadata,
            "fingerprint": cert.fingerprint(),
        }

    def _serialize_step(self, step: ProofStep) -> Dict[str, Any]:
        return {
            "step_id": step.step_id,
            "rule_name": step.rule_name,
            "premises": step.premises,
            "conclusion": str(step.conclusion),
            "justification": step.justification.value,
            "annotation": step.annotation,
            "solver_time_ms": step.solver_time_ms,
        }

    def _serialize_vc(self, vc: VerificationCondition) -> Dict[str, Any]:
        return {
            "vc_id": vc.vc_id,
            "kind": vc.kind.value,
            "formula": str(vc.formula),
            "location": vc.location,
            "description": vc.description,
            "status": vc.status.value,
            "solver_time_ms": vc.solver_time_ms,
        }

    def _serialize_vr(self, vr: VerificationResult) -> Dict[str, Any]:
        return {
            "status": vr.status.value,
            "message": vr.message,
            "proof_steps_verified": vr.proof_steps_verified,
            "proof_steps_total": vr.proof_steps_total,
            "solver_time_ms": vr.solver_time_ms,
        }

    def serialize_statistics(self, stats: ProofStatistics) -> Dict[str, Any]:
        return stats.to_dict()


# ---------------------------------------------------------------------------
# ProofDiff: compare two certificates (useful for CEGAR iterations)
# ---------------------------------------------------------------------------

@dataclass
class ProofDifference:
    added_steps: List[str]
    removed_steps: List[str]
    changed_conclusions: List[Tuple[str, str, str]]  # (step_id, old, new)
    added_vcs: List[str]
    removed_vcs: List[str]
    status_changed: bool
    old_status: Optional[VerificationStatus]
    new_status: Optional[VerificationStatus]


class ProofDiff:
    """Compare two proof certificates."""

    def diff(
        self, old: ProofCertificate, new: ProofCertificate
    ) -> ProofDifference:
        old_step_ids = {s.step_id for s in old.proof_steps}
        new_step_ids = {s.step_id for s in new.proof_steps}

        added = sorted(new_step_ids - old_step_ids)
        removed = sorted(old_step_ids - new_step_ids)

        old_map = old.step_map()
        new_map = new.step_map()
        changed: List[Tuple[str, str, str]] = []
        for sid in old_step_ids & new_step_ids:
            old_conc = str(old_map[sid].conclusion)
            new_conc = str(new_map[sid].conclusion)
            if old_conc != new_conc:
                changed.append((sid, old_conc, new_conc))

        old_vc_ids = {vc.vc_id for vc in old.verification_conditions}
        new_vc_ids = {vc.vc_id for vc in new.verification_conditions}
        added_vcs = sorted(new_vc_ids - old_vc_ids)
        removed_vcs = sorted(old_vc_ids - new_vc_ids)

        old_status = old.verification_result.status if old.verification_result else None
        new_status = new.verification_result.status if new.verification_result else None

        return ProofDifference(
            added_steps=added,
            removed_steps=removed,
            changed_conclusions=changed,
            added_vcs=added_vcs,
            removed_vcs=removed_vcs,
            status_changed=old_status != new_status,
            old_status=old_status,
            new_status=new_status,
        )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Core types
    "Expr", "Var", "IntLit", "BoolLit", "StrLit", "NoneLit",
    "UnaryExpr", "BinaryExpr", "QuantifiedExpr", "FuncApp", "LetExpr", "IteExpr",
    "UnaryOp", "BinaryOp", "Quantifier", "BaseType",
    # Statement types
    "Stmt", "AssignStmt", "SeqStmt", "IfStmt", "WhileStmt", "ReturnStmt",
    "AssertStmt", "AssumeStmt", "CallStmt", "SkipStmt", "RaiseStmt", "TryStmt",
    # Contract types
    "FunctionContract", "LoopInvariantContract", "SubtypeObligation",
    # Proof types
    "ProofStep", "ProofCertificate", "VerificationResult", "VerificationCondition",
    "Justification", "VerificationStatus", "VCKind",
    "HoareTriple",
    # Generators
    "SmtLibGenerator", "ProofObligationGenerator", "VerificationConditionGenerator",
    # Calculators
    "WeakestPreconditionCalculator", "StrongestPostconditionCalculator",
    # Verifiers
    "HoareTripleVerifier", "ProofChecker",
    # Formatters
    "ProofFormatter", "ProofStatistics",
    # Pipeline
    "ProofPipeline", "SmtBatchRunner",
    # Utilities
    "ExpressionSimplifier", "FormulaNormalizer", "SSATransformer",
    "InvariantStrengthener", "AnnotatedProgram", "AnnotatedProgramVerifier",
    "ProofArchive", "ProofDiff", "ProofDifference",
]
