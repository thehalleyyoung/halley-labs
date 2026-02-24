from __future__ import annotations

import abc
import enum
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


logger = logging.getLogger(__name__)


# ===================================================================
# Local type definitions (no cross-module imports)
# ===================================================================

class Sort(enum.Enum):
    INT = "Int"
    BOOL = "Bool"
    TAG = "Tag"
    STR = "Str"


class ComparisonOp(enum.Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class ArithOp(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


class UnaryArithOp(enum.Enum):
    NEG = "neg"
    ABS = "abs"


# -- Expression nodes --

@dataclass(frozen=True)
class Var:
    name: str
    sort: Sort = Sort.INT
    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.name})

@dataclass(frozen=True)
class Const:
    value: Union[int, bool, str]
    sort: Sort = Sort.INT
    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

@dataclass(frozen=True)
class Len:
    arg: Expr
    def free_vars(self) -> FrozenSet[str]:
        return self.arg.free_vars()

@dataclass(frozen=True)
class BinOp:
    op: ArithOp
    left: Expr
    right: Expr
    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

@dataclass(frozen=True)
class UnaryOp:
    op: UnaryArithOp
    operand: Expr
    def free_vars(self) -> FrozenSet[str]:
        return self.operand.free_vars()

Expr = Union[Var, Const, Len, BinOp, UnaryOp]


# -- Predicate nodes --

@dataclass(frozen=True)
class Comparison:
    op: ComparisonOp
    left: Expr
    right: Expr
    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

@dataclass(frozen=True)
class IsInstance:
    var: str
    tag: str
    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

@dataclass(frozen=True)
class IsNone:
    var: str
    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

@dataclass(frozen=True)
class IsTruthy:
    var: str
    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

@dataclass(frozen=True)
class HasAttr:
    var: str
    key: str
    def free_vars(self) -> FrozenSet[str]:
        return frozenset({self.var})

@dataclass(frozen=True)
class And:
    conjuncts: Tuple[Predicate, ...]
    def free_vars(self) -> FrozenSet[str]:
        r: FrozenSet[str] = frozenset()
        for c in self.conjuncts:
            r = r | c.free_vars()
        return r

@dataclass(frozen=True)
class Or:
    disjuncts: Tuple[Predicate, ...]
    def free_vars(self) -> FrozenSet[str]:
        r: FrozenSet[str] = frozenset()
        for d in self.disjuncts:
            r = r | d.free_vars()
        return r

@dataclass(frozen=True)
class Not:
    operand: Predicate
    def free_vars(self) -> FrozenSet[str]:
        return self.operand.free_vars()

@dataclass(frozen=True)
class Implies:
    antecedent: Predicate
    consequent: Predicate
    def free_vars(self) -> FrozenSet[str]:
        return self.antecedent.free_vars() | self.consequent.free_vars()

@dataclass(frozen=True)
class Iff:
    left: Predicate
    right: Predicate
    def free_vars(self) -> FrozenSet[str]:
        return self.left.free_vars() | self.right.free_vars()

@dataclass(frozen=True)
class BoolLit:
    value: bool
    def free_vars(self) -> FrozenSet[str]:
        return frozenset()

Predicate = Union[
    Comparison, IsInstance, IsNone, IsTruthy, HasAttr,
    And, Or, Not, Implies, Iff, BoolLit,
]


# -- Refinement types --

@dataclass(frozen=True)
class BaseType:
    tag: str  # "int", "str", "bool", "float", "list", "dict", "NoneType", ...

@dataclass(frozen=True)
class RefinedType:
    base: BaseType
    var: str
    refinement: Predicate

@dataclass(frozen=True)
class FunctionType:
    param_names: Tuple[str, ...]
    param_types: Tuple[RefinementType, ...]
    return_type: RefinementType

@dataclass(frozen=True)
class UnionType:
    alternatives: Tuple[RefinementType, ...]

@dataclass(frozen=True)
class RecursiveType:
    name: str
    body: RefinementType

RefinementType = Union[BaseType, RefinedType, FunctionType, UnionType, RecursiveType]


# -- CFG path (for counterexample representation) --

@dataclass
class PathEdge:
    source: int
    target: int
    condition: Optional[Predicate] = None
    assignment: Optional[Tuple[str, Expr]] = None
    label: str = ""

@dataclass
class CfgPath:
    edges: List[PathEdge] = field(default_factory=list)
    entry_state: Dict[str, Any] = field(default_factory=dict)

    def add_edge(self, edge: PathEdge) -> None:
        self.edges.append(edge)

    @property
    def length(self) -> int:
        return len(self.edges)


# -- Typing context --

@dataclass
class TypingContext:
    bindings: Dict[str, RefinementType] = field(default_factory=dict)
    assumptions: List[Predicate] = field(default_factory=list)

    def add_binding(self, name: str, ty: RefinementType) -> None:
        self.bindings[name] = ty

    def add_assumption(self, pred: Predicate) -> None:
        self.assumptions.append(pred)

    def get_binding(self, name: str) -> Optional[RefinementType]:
        return self.bindings.get(name)


# ===================================================================
# SMT solver result types
# ===================================================================

class SatResult(enum.Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class SmtModel:
    """A satisfying assignment from the SMT solver."""
    variable_values: Dict[str, Any] = field(default_factory=dict)
    function_interpretations: Dict[str, Any] = field(default_factory=dict)

    def get_int(self, name: str) -> Optional[int]:
        v = self.variable_values.get(name)
        if isinstance(v, int):
            return v
        return None

    def get_bool(self, name: str) -> Optional[bool]:
        v = self.variable_values.get(name)
        if isinstance(v, bool):
            return v
        return None

    def get_str(self, name: str) -> Optional[str]:
        v = self.variable_values.get(name)
        if isinstance(v, str):
            return v
        return None

    def get_tag(self, name: str) -> Optional[str]:
        v = self.variable_values.get(name)
        if isinstance(v, str):
            return v
        return None

    def __repr__(self) -> str:
        items = [f"{k}={v!r}" for k, v in sorted(self.variable_values.items())]
        return f"Model({', '.join(items)})"


@dataclass
class SmtUnsatCore:
    """An unsatisfiable core — a minimal set of assertions that are unsat."""
    core: List[Predicate] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.core)

    def __repr__(self) -> str:
        return f"UnsatCore({len(self.core)} formulas)"


@dataclass
class SmtQueryResult:
    """Full result of an SMT query."""
    result: SatResult
    model: Optional[SmtModel] = None
    unsat_core: Optional[SmtUnsatCore] = None
    time_seconds: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# SmtSolver — abstract interface
# ===================================================================

class SmtSolver(abc.ABC):
    """Abstract interface for SMT solvers."""

    @abc.abstractmethod
    def check_sat(self) -> SatResult:
        """Check satisfiability of the current assertion stack."""
        ...

    @abc.abstractmethod
    def check_sat_assuming(self, assumptions: List[Predicate]) -> SatResult:
        """Check satisfiability under additional assumptions."""
        ...

    @abc.abstractmethod
    def get_model(self) -> Optional[SmtModel]:
        """Get a satisfying model (only valid after SAT result)."""
        ...

    @abc.abstractmethod
    def get_unsat_core(self) -> Optional[SmtUnsatCore]:
        """Get an unsatisfiable core (only valid after UNSAT result)."""
        ...

    @abc.abstractmethod
    def push(self) -> None:
        """Push a new scope on the assertion stack."""
        ...

    @abc.abstractmethod
    def pop(self, n: int = 1) -> None:
        """Pop *n* scopes from the assertion stack."""
        ...

    @abc.abstractmethod
    def assert_formula(self, formula: Predicate, label: Optional[str] = None) -> None:
        """Add a formula to the current assertion stack."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the solver to its initial state."""
        ...

    @abc.abstractmethod
    def set_timeout(self, milliseconds: int) -> None:
        """Set a timeout for subsequent queries."""
        ...

    @abc.abstractmethod
    def set_logic(self, logic: str) -> None:
        """Set the SMT logic (e.g. QF_UFLIA)."""
        ...

    def declare_int(self, name: str) -> None:
        """Declare an integer variable."""
        pass

    def declare_bool(self, name: str) -> None:
        """Declare a boolean variable."""
        pass

    def declare_tag(self, name: str, domain: Optional[List[str]] = None) -> None:
        """Declare a type-tag variable."""
        pass

    def declare_str(self, name: str) -> None:
        """Declare a string variable."""
        pass


# ===================================================================
# Z3Solver — Z3 Python API wrapper
# ===================================================================

class Z3SolverError(Exception):
    """Error raised by Z3 solver operations."""
    pass


class Z3Solver(SmtSolver):
    """Full Z3 integration for QF_UFLIA + type tags.

    Wraps the z3-solver Python package.  If z3 is not installed, the
    constructor raises ``ImportError``; callers should fall back to
    ``FallbackSolver``.
    """

    # Standard type tag domain
    DEFAULT_TAGS: List[str] = [
        "int", "float", "str", "bool", "list", "tuple", "dict", "set",
        "bytes", "NoneType", "complex", "frozenset", "bytearray",
        "memoryview", "range", "type", "function", "object",
    ]

    def __init__(
        self,
        *,
        timeout_ms: int = 30000,
        logic: str = "QF_UFLIA",
        tag_domain: Optional[List[str]] = None,
    ) -> None:
        try:
            import z3 as _z3
        except ImportError:
            raise ImportError(
                "z3-solver package is required for Z3Solver. "
                "Install with: pip install z3-solver"
            )
        self._z3 = _z3
        self._solver = _z3.Solver()
        self._solver.set("timeout", timeout_ms)
        self._timeout_ms = timeout_ms
        self._logic = logic

        # Sort declarations
        self._int_sort = _z3.IntSort()
        self._bool_sort = _z3.BoolSort()
        self._str_sort = _z3.StringSort()

        # Tag sort — finite enum (unique name to avoid Z3 redeclaration errors)
        import uuid
        self._tag_names = tag_domain or list(self.DEFAULT_TAGS)
        tag_sort_name = f"Tag_{uuid.uuid4().hex[:8]}"
        self._tag_sort, self._tag_constructors = _z3.EnumSort(
            tag_sort_name, self._tag_names
        )
        self._tag_map: Dict[str, Any] = {
            name: ctor for name, ctor in zip(self._tag_names, self._tag_constructors)
        }

        # Declared variables
        self._int_vars: Dict[str, Any] = {}
        self._bool_vars: Dict[str, Any] = {}
        self._tag_vars: Dict[str, Any] = {}
        self._str_vars: Dict[str, Any] = {}

        # Function declarations
        self._len_fn = _z3.Function("len", self._int_sort, self._int_sort)
        self._isinstance_fn = _z3.Function(
            "isinstance_fn", self._int_sort, self._tag_sort, self._bool_sort
        )
        self._is_none_fn = _z3.Function("is_none_fn", self._int_sort, self._bool_sort)
        self._is_truthy_fn = _z3.Function(
            "is_truthy_fn", self._int_sort, self._bool_sort
        )
        self._hasattr_fn = _z3.Function(
            "hasattr_fn", self._int_sort, self._str_sort, self._bool_sort
        )

        # typeof function: maps a value-variable to its tag
        self._typeof_fn = _z3.Function("typeof", self._int_sort, self._tag_sort)

        # Track labels for unsat core
        self._labels: Dict[str, Any] = {}
        self._label_counter = 0

        # Last result
        self._last_result: Optional[SatResult] = None

        # Axioms
        self._axioms_added = False

    # -- scope management --------------------------------------------------

    def push(self) -> None:
        self._solver.push()

    def pop(self, n: int = 1) -> None:
        self._solver.pop(n)

    def reset(self) -> None:
        self._solver.reset()
        self._int_vars.clear()
        self._bool_vars.clear()
        self._tag_vars.clear()
        self._str_vars.clear()
        self._labels.clear()
        self._label_counter = 0
        self._last_result = None
        self._axioms_added = False

    def set_timeout(self, milliseconds: int) -> None:
        self._timeout_ms = milliseconds
        self._solver.set("timeout", milliseconds)

    def set_logic(self, logic: str) -> None:
        self._logic = logic

    # -- variable declarations ---------------------------------------------

    def declare_int(self, name: str) -> None:
        if name not in self._int_vars:
            self._int_vars[name] = self._z3.Int(name)

    def declare_bool(self, name: str) -> None:
        if name not in self._bool_vars:
            self._bool_vars[name] = self._z3.Bool(name)

    def declare_tag(self, name: str, domain: Optional[List[str]] = None) -> None:
        if name not in self._tag_vars:
            self._tag_vars[name] = self._z3.Const(name, self._tag_sort)

    def declare_str(self, name: str) -> None:
        if name not in self._str_vars:
            self._str_vars[name] = self._z3.String(name)

    def _get_or_declare_int(self, name: str) -> Any:
        if name not in self._int_vars:
            self._int_vars[name] = self._z3.Int(name)
        return self._int_vars[name]

    def _get_or_declare_bool(self, name: str) -> Any:
        if name not in self._bool_vars:
            self._bool_vars[name] = self._z3.Bool(name)
        return self._bool_vars[name]

    def _get_or_declare_tag(self, name: str) -> Any:
        if name not in self._tag_vars:
            self._tag_vars[name] = self._z3.Const(name, self._tag_sort)
        return self._tag_vars[name]

    # -- axiom generation --------------------------------------------------

    def _ensure_axioms(self) -> None:
        """Add axioms for function interpretations."""
        if self._axioms_added:
            return
        self._axioms_added = True
        z3 = self._z3

        x = z3.Int("__ax_x")
        # len(x) >= 0
        self._solver.add(z3.ForAll([x], self._len_fn(x) >= 0))

        # is_none(x) => typeof(x) == NoneType
        if "NoneType" in self._tag_map:
            self._solver.add(
                z3.ForAll(
                    [x],
                    z3.Implies(
                        self._is_none_fn(x),
                        self._typeof_fn(x) == self._tag_map["NoneType"],
                    ),
                )
            )

        # isinstance(x, T) <=> typeof(x) == T  (for concrete tags)
        for tag_name, tag_val in self._tag_map.items():
            self._solver.add(
                z3.ForAll(
                    [x],
                    self._isinstance_fn(x, tag_val) == (
                        self._typeof_fn(x) == tag_val
                    ),
                )
            )

    # -- assert formula ----------------------------------------------------

    def assert_formula(self, formula: Predicate, label: Optional[str] = None) -> None:
        self._ensure_axioms()
        z3_formula = self._encode_predicate(formula)
        if label is not None:
            lbl = self._z3.Bool(label)
            self._labels[label] = lbl
            self._solver.assert_and_track(z3_formula, lbl)
        else:
            self._solver.add(z3_formula)

    # -- check sat ---------------------------------------------------------

    def check_sat(self) -> SatResult:
        self._ensure_axioms()
        start = time.monotonic()
        result = self._solver.check()
        elapsed = time.monotonic() - start
        self._last_result = self._translate_result(result)
        logger.debug("Z3 check_sat: %s (%.3fs)", self._last_result.value, elapsed)
        return self._last_result

    def check_sat_assuming(self, assumptions: List[Predicate]) -> SatResult:
        self._ensure_axioms()
        z3_assumptions = [self._encode_predicate(a) for a in assumptions]
        start = time.monotonic()
        result = self._solver.check(*z3_assumptions)
        elapsed = time.monotonic() - start
        self._last_result = self._translate_result(result)
        logger.debug(
            "Z3 check_sat_assuming: %s (%.3fs)", self._last_result.value, elapsed
        )
        return self._last_result

    # -- model extraction --------------------------------------------------

    def get_model(self) -> Optional[SmtModel]:
        if self._last_result != SatResult.SAT:
            return None
        z3_model = self._solver.model()
        return self._extract_model(z3_model)

    def _extract_model(self, z3_model: Any) -> SmtModel:
        values: Dict[str, Any] = {}
        for name, var in self._int_vars.items():
            val = z3_model.eval(var, model_completion=True)
            try:
                values[name] = val.as_long()
            except Exception:
                values[name] = str(val)
        for name, var in self._bool_vars.items():
            val = z3_model.eval(var, model_completion=True)
            values[name] = bool(val)
        for name, var in self._tag_vars.items():
            val = z3_model.eval(var, model_completion=True)
            values[name] = str(val)

        func_interps: Dict[str, Any] = {}
        for decl in z3_model.decls():
            if decl.arity() > 0:
                interp = z3_model.get_interp(decl)
                if interp is not None:
                    func_interps[decl.name()] = str(interp)

        return SmtModel(
            variable_values=values,
            function_interpretations=func_interps,
        )

    # -- unsat core --------------------------------------------------------

    def get_unsat_core(self) -> Optional[SmtUnsatCore]:
        if self._last_result != SatResult.UNSAT:
            return None
        z3_core = self._solver.unsat_core()
        labels = [str(c) for c in z3_core]
        return SmtUnsatCore(core=[], labels=labels)

    # -- encoding predicates → Z3 -----------------------------------------

    def _encode_predicate(self, pred: Predicate) -> Any:
        z3 = self._z3
        if isinstance(pred, BoolLit):
            return z3.BoolVal(pred.value)
        if isinstance(pred, Comparison):
            left = self._encode_expr(pred.left)
            right = self._encode_expr(pred.right)
            return self._encode_comparison_op(pred.op, left, right)
        if isinstance(pred, IsInstance):
            var = self._get_or_declare_int(pred.var)
            tag_val = self._tag_map.get(pred.tag)
            if tag_val is None:
                logger.warning("Unknown tag %r, treating as false", pred.tag)
                return z3.BoolVal(False)
            return self._isinstance_fn(var, tag_val)
        if isinstance(pred, IsNone):
            var = self._get_or_declare_int(pred.var)
            return self._is_none_fn(var)
        if isinstance(pred, IsTruthy):
            var = self._get_or_declare_int(pred.var)
            return self._is_truthy_fn(var)
        if isinstance(pred, HasAttr):
            var = self._get_or_declare_int(pred.var)
            key = z3.StringVal(pred.key)
            return self._hasattr_fn(var, key)
        if isinstance(pred, And):
            if not pred.conjuncts:
                return z3.BoolVal(True)
            encoded = [self._encode_predicate(c) for c in pred.conjuncts]
            return z3.And(*encoded)
        if isinstance(pred, Or):
            if not pred.disjuncts:
                return z3.BoolVal(False)
            encoded = [self._encode_predicate(d) for d in pred.disjuncts]
            return z3.Or(*encoded)
        if isinstance(pred, Not):
            return z3.Not(self._encode_predicate(pred.operand))
        if isinstance(pred, Implies):
            return z3.Implies(
                self._encode_predicate(pred.antecedent),
                self._encode_predicate(pred.consequent),
            )
        if isinstance(pred, Iff):
            l = self._encode_predicate(pred.left)
            r = self._encode_predicate(pred.right)
            return l == r
        raise Z3SolverError(f"Cannot encode predicate: {type(pred)}")

    def _encode_expr(self, expr: Expr) -> Any:
        z3 = self._z3
        if isinstance(expr, Var):
            return self._get_or_declare_int(expr.name)
        if isinstance(expr, Const):
            if isinstance(expr.value, bool):
                return z3.BoolVal(expr.value)
            if isinstance(expr.value, int):
                return z3.IntVal(expr.value)
            if isinstance(expr.value, str):
                return z3.StringVal(expr.value)
            return z3.IntVal(0)
        if isinstance(expr, Len):
            inner = self._encode_expr(expr.arg)
            return self._len_fn(inner)
        if isinstance(expr, BinOp):
            left = self._encode_expr(expr.left)
            right = self._encode_expr(expr.right)
            if expr.op == ArithOp.ADD:
                return left + right
            if expr.op == ArithOp.SUB:
                return left - right
            if expr.op == ArithOp.MUL:
                return left * right
            if expr.op == ArithOp.DIV:
                return left / right
            if expr.op == ArithOp.MOD:
                return left % right
        if isinstance(expr, UnaryOp):
            inner = self._encode_expr(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return -inner
            if expr.op == UnaryArithOp.ABS:
                return z3.If(inner >= 0, inner, -inner)
        raise Z3SolverError(f"Cannot encode expression: {type(expr)}")

    def _encode_comparison_op(self, op: ComparisonOp, left: Any, right: Any) -> Any:
        if op == ComparisonOp.EQ:
            return left == right
        if op == ComparisonOp.NE:
            return left != right
        if op == ComparisonOp.LT:
            return left < right
        if op == ComparisonOp.LE:
            return left <= right
        if op == ComparisonOp.GT:
            return left > right
        if op == ComparisonOp.GE:
            return left >= right
        raise Z3SolverError(f"Unknown comparison op: {op}")

    def _translate_result(self, z3_result: Any) -> SatResult:
        z3 = self._z3
        if z3_result == z3.sat:
            return SatResult.SAT
        if z3_result == z3.unsat:
            return SatResult.UNSAT
        return SatResult.UNKNOWN

    # -- optimization (minimize / maximize) --------------------------------

    def minimize(self, expr: Expr) -> Optional[SmtQueryResult]:
        """Find the minimum value of *expr* subject to current assertions."""
        z3 = self._z3
        opt = z3.Optimize()
        for assertion in self._solver.assertions():
            opt.add(assertion)
        z3_expr = self._encode_expr(expr)
        opt.minimize(z3_expr)
        start = time.monotonic()
        result = opt.check()
        elapsed = time.monotonic() - start
        sat_result = self._translate_result(result)
        if sat_result == SatResult.SAT:
            model = self._extract_model(opt.model())
            return SmtQueryResult(result=sat_result, model=model, time_seconds=elapsed)
        return SmtQueryResult(result=sat_result, time_seconds=elapsed)

    def maximize(self, expr: Expr) -> Optional[SmtQueryResult]:
        """Find the maximum value of *expr* subject to current assertions."""
        z3 = self._z3
        opt = z3.Optimize()
        for assertion in self._solver.assertions():
            opt.add(assertion)
        z3_expr = self._encode_expr(expr)
        opt.maximize(z3_expr)
        start = time.monotonic()
        result = opt.check()
        elapsed = time.monotonic() - start
        sat_result = self._translate_result(result)
        if sat_result == SatResult.SAT:
            model = self._extract_model(opt.model())
            return SmtQueryResult(result=sat_result, model=model, time_seconds=elapsed)
        return SmtQueryResult(result=sat_result, time_seconds=elapsed)

    # -- quantifier elimination --------------------------------------------

    def quantifier_eliminate(self, formula: Predicate) -> Optional[Predicate]:
        """Apply quantifier elimination via Z3 tactics."""
        z3 = self._z3
        z3_formula = self._encode_predicate(formula)
        try:
            tactic = z3.Then("qe", "simplify")
            goal = z3.Goal()
            goal.add(z3_formula)
            result = tactic(goal)
            if len(result) == 1 and len(result[0]) == 1:
                return BoolLit(True)  # simplified away
            return formula  # couldn't fully eliminate
        except Exception:
            return None

    # -- interpolant computation -------------------------------------------

    def compute_interpolant(
        self, a: Predicate, b: Predicate
    ) -> Optional[Predicate]:
        """Compute a predicate I such that A ⊨ I and I ∧ B is unsat.

        Returns I or None on failure.  Uses Z3 satisfiability checking.
        Note: the result may not satisfy Craig's vocabulary restriction.
        """
        z3 = self._z3
        z3_a = self._encode_predicate(a)
        z3_b = self._encode_predicate(b)

        # Check if A ∧ B is unsatisfiable; if so, return A as interpolant
        try:
            s = z3.Solver()
            s.set("timeout", self._timeout_ms)
            s.add(z3_a)
            s.add(z3_b)
            if s.check() == z3.unsat:
                # A is a valid interpolant: A ⊨ A, and A ∧ B is unsat
                return a
            return None
        except Exception:
            return None

    # -- SMT-LIB generation ------------------------------------------------

    def to_smt_lib(self) -> str:
        """Generate SMT-LIB2 representation of the current state."""
        return self._solver.to_smt2()

    def sexpr(self) -> str:
        """Get s-expression representation."""
        return self._solver.sexpr()

    # -- statistics --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """Get solver statistics."""
        stats = self._solver.statistics()
        result: Dict[str, Any] = {}
        for key in stats.keys():
            result[key] = stats[key]
        return result


# ===================================================================
# FallbackSolver — lightweight solver when Z3 is not available
# ===================================================================

@dataclass
class _Interval:
    """Simple integer interval [lo, hi)."""
    lo: Optional[int] = None
    hi: Optional[int] = None

    @property
    def is_empty(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo >= self.hi
        return False

    def contains(self, v: int) -> bool:
        if self.lo is not None and v < self.lo:
            return False
        if self.hi is not None and v >= self.hi:
            return False
        return True

    def intersect(self, other: _Interval) -> _Interval:
        lo = max(self.lo, other.lo) if self.lo is not None and other.lo is not None else (self.lo or other.lo)
        hi = min(self.hi, other.hi) if self.hi is not None and other.hi is not None else (self.hi or other.hi)
        return _Interval(lo, hi)


class FallbackSolver(SmtSolver):
    """Lightweight solver using interval analysis and constraint propagation.

    Does not require Z3. May return ``UNKNOWN`` for complex formulas.
    """

    def __init__(self, *, timeout_ms: int = 5000) -> None:
        self._timeout_ms = timeout_ms
        self._assertions: List[List[Predicate]] = [[]]
        self._last_result: Optional[SatResult] = None
        self._last_model: Optional[SmtModel] = None
        self._logic = "QF_UFLIA"
        self._declared_ints: Set[str] = set()
        self._declared_bools: Set[str] = set()
        self._declared_tags: Dict[str, Optional[List[str]]] = {}
        self._declared_strs: Set[str] = set()
        self._label_map: Dict[str, Predicate] = {}

    # -- scope management --------------------------------------------------

    def push(self) -> None:
        self._assertions.append([])

    def pop(self, n: int = 1) -> None:
        for _ in range(min(n, len(self._assertions) - 1)):
            self._assertions.pop()

    def reset(self) -> None:
        self._assertions = [[]]
        self._last_result = None
        self._last_model = None
        self._declared_ints.clear()
        self._declared_bools.clear()
        self._declared_tags.clear()
        self._declared_strs.clear()
        self._label_map.clear()

    def set_timeout(self, milliseconds: int) -> None:
        self._timeout_ms = milliseconds

    def set_logic(self, logic: str) -> None:
        self._logic = logic

    def declare_int(self, name: str) -> None:
        self._declared_ints.add(name)

    def declare_bool(self, name: str) -> None:
        self._declared_bools.add(name)

    def declare_tag(self, name: str, domain: Optional[List[str]] = None) -> None:
        self._declared_tags[name] = domain

    def declare_str(self, name: str) -> None:
        self._declared_strs.add(name)

    def assert_formula(self, formula: Predicate, label: Optional[str] = None) -> None:
        self._assertions[-1].append(formula)
        if label is not None:
            self._label_map[label] = formula

    def _all_assertions(self) -> List[Predicate]:
        result: List[Predicate] = []
        for scope in self._assertions:
            result.extend(scope)
        return result

    # -- check sat ---------------------------------------------------------

    def check_sat(self) -> SatResult:
        all_preds = self._all_assertions()
        start = time.monotonic()
        result = self._check_satisfiability(all_preds)
        elapsed = time.monotonic() - start
        self._last_result = result
        logger.debug("Fallback check_sat: %s (%.3fs)", result.value, elapsed)
        return result

    def check_sat_assuming(self, assumptions: List[Predicate]) -> SatResult:
        all_preds = self._all_assertions() + list(assumptions)
        start = time.monotonic()
        result = self._check_satisfiability(all_preds)
        elapsed = time.monotonic() - start
        self._last_result = result
        return result

    def get_model(self) -> Optional[SmtModel]:
        return self._last_model

    def get_unsat_core(self) -> Optional[SmtUnsatCore]:
        if self._last_result == SatResult.UNSAT:
            return SmtUnsatCore(core=self._all_assertions())
        return None

    # -- internal solving --------------------------------------------------

    def _check_satisfiability(self, preds: List[Predicate]) -> SatResult:
        """Attempt to check satisfiability using interval analysis."""
        if not preds:
            self._last_model = SmtModel()
            return SatResult.SAT

        # Collect free variables
        all_vars: Set[str] = set()
        for p in preds:
            all_vars |= set(p.free_vars())

        # Extract intervals
        intervals: Dict[str, _Interval] = {v: _Interval() for v in all_vars}
        type_constraints: Dict[str, Set[str]] = {}
        none_constraints: Dict[str, Optional[bool]] = {}
        bool_constraints: Dict[str, Optional[bool]] = {}

        for p in preds:
            if isinstance(p, BoolLit):
                if not p.value:
                    self._last_model = None
                    return SatResult.UNSAT
                continue

            if isinstance(p, Comparison):
                self._process_comparison(p, intervals)
            elif isinstance(p, IsInstance):
                if p.var not in type_constraints:
                    type_constraints[p.var] = set()
                type_constraints[p.var].add(p.tag)
            elif isinstance(p, IsNone):
                none_constraints[p.var] = True
            elif isinstance(p, Not):
                if isinstance(p.operand, IsNone):
                    none_constraints[p.operand.var] = False
                elif isinstance(p.operand, IsInstance):
                    pass  # negative type constraint
                elif isinstance(p.operand, Comparison):
                    self._process_negated_comparison(p.operand, intervals)
                else:
                    return SatResult.UNKNOWN
            elif isinstance(p, And):
                # Flatten and recurse
                result = self._check_satisfiability(list(p.conjuncts))
                if result != SatResult.SAT:
                    return result
            elif isinstance(p, Or):
                # Try each disjunct
                any_sat = False
                for d in p.disjuncts:
                    sub_result = self._check_satisfiability([d])
                    if sub_result == SatResult.SAT:
                        any_sat = True
                        break
                if not any_sat:
                    return SatResult.UNKNOWN
            else:
                return SatResult.UNKNOWN

        # Check interval emptiness
        for var, iv in intervals.items():
            if iv.is_empty:
                self._last_model = None
                return SatResult.UNSAT

        # Check type constraint conflicts
        for var, is_none in none_constraints.items():
            if is_none and var in type_constraints:
                tags = type_constraints[var]
                if tags and "NoneType" not in tags:
                    self._last_model = None
                    return SatResult.UNSAT

        # Build model
        model_values: Dict[str, Any] = {}
        for var in all_vars:
            iv = intervals.get(var, _Interval())
            if iv.lo is not None:
                model_values[var] = iv.lo
            elif iv.hi is not None:
                model_values[var] = iv.hi - 1
            else:
                model_values[var] = 0

        self._last_model = SmtModel(variable_values=model_values)
        return SatResult.SAT

    def _process_comparison(
        self, cmp: Comparison, intervals: Dict[str, _Interval]
    ) -> None:
        """Tighten intervals based on a comparison."""
        if isinstance(cmp.left, Var) and isinstance(cmp.right, Const):
            var = cmp.left.name
            val = cmp.right.value
            if not isinstance(val, int):
                return
            if var not in intervals:
                intervals[var] = _Interval()
            iv = intervals[var]
            if cmp.op == ComparisonOp.EQ:
                iv.lo = max(iv.lo, val) if iv.lo is not None else val
                iv.hi = min(iv.hi, val + 1) if iv.hi is not None else val + 1
            elif cmp.op == ComparisonOp.NE:
                # Can't express in a single interval; skip
                pass
            elif cmp.op == ComparisonOp.LT:
                iv.hi = min(iv.hi, val) if iv.hi is not None else val
            elif cmp.op == ComparisonOp.LE:
                iv.hi = min(iv.hi, val + 1) if iv.hi is not None else val + 1
            elif cmp.op == ComparisonOp.GT:
                iv.lo = max(iv.lo, val + 1) if iv.lo is not None else val + 1
            elif cmp.op == ComparisonOp.GE:
                iv.lo = max(iv.lo, val) if iv.lo is not None else val

        elif isinstance(cmp.right, Var) and isinstance(cmp.left, Const):
            # c op x  →  x flipped_op c
            flipped = {
                ComparisonOp.EQ: ComparisonOp.EQ,
                ComparisonOp.NE: ComparisonOp.NE,
                ComparisonOp.LT: ComparisonOp.GT,
                ComparisonOp.LE: ComparisonOp.GE,
                ComparisonOp.GT: ComparisonOp.LT,
                ComparisonOp.GE: ComparisonOp.LE,
            }
            flipped_cmp = Comparison(flipped[cmp.op], cmp.right, cmp.left)
            self._process_comparison(flipped_cmp, intervals)

    def _process_negated_comparison(
        self, cmp: Comparison, intervals: Dict[str, _Interval]
    ) -> None:
        """Tighten intervals based on negation of a comparison."""
        negated_op = {
            ComparisonOp.EQ: ComparisonOp.NE,
            ComparisonOp.NE: ComparisonOp.EQ,
            ComparisonOp.LT: ComparisonOp.GE,
            ComparisonOp.LE: ComparisonOp.GT,
            ComparisonOp.GT: ComparisonOp.LE,
            ComparisonOp.GE: ComparisonOp.LT,
        }
        neg_cmp = Comparison(negated_op[cmp.op], cmp.left, cmp.right)
        self._process_comparison(neg_cmp, intervals)


# ===================================================================
# SmtEncoder — encode refinement typing problems as SMT
# ===================================================================

class SmtEncoder:
    """Encode refinement typing problems as SMT formulas."""

    def __init__(self) -> None:
        self._fresh_counter = 0

    def fresh_var(self, prefix: str = "_v") -> str:
        self._fresh_counter += 1
        return f"{prefix}_{self._fresh_counter}"

    def encode_subtype_check(
        self,
        ty1: RefinementType,
        ty2: RefinementType,
        context: TypingContext,
    ) -> Predicate:
        """Encode a subtype check {x:τ₁|φ₁} <: {x:τ₂|φ₂} as an SMT formula.

        The resulting formula is valid iff the subtype relation holds.
        We check: Γ, φ₁ ⊨ φ₂ (under the same variable).
        """
        if isinstance(ty1, RefinedType) and isinstance(ty2, RefinedType):
            if ty1.base != ty2.base:
                # Base types must match (or be subtypes)
                if not self._base_subtype(ty1.base, ty2.base):
                    return BoolLit(False)

            # Substitute ty2's binder with ty1's binder
            phi1 = ty1.refinement
            phi2 = ty2.refinement
            if ty1.var != ty2.var:
                phi2 = self._rename_in_pred(phi2, ty2.var, ty1.var)

            # Build context assumptions
            context_pred = self._context_to_pred(context)

            # Check: context ∧ φ₁ ⊨ φ₂
            # Encoded as: ¬(context ∧ φ₁ ∧ ¬φ₂) should be valid (unsat)
            return And((context_pred, phi1, Not(phi2)))

        if isinstance(ty1, BaseType) and isinstance(ty2, BaseType):
            if self._base_subtype(ty1, ty2):
                return BoolLit(False)  # trivially valid → negation is unsat
            return BoolLit(True)  # not a subtype → negation is sat

        if isinstance(ty1, FunctionType) and isinstance(ty2, FunctionType):
            return self._encode_function_subtype(ty1, ty2, context)

        if isinstance(ty1, UnionType):
            # Each alternative of ty1 must be a subtype of ty2
            conjuncts = [
                self.encode_subtype_check(alt, ty2, context)
                for alt in ty1.alternatives
            ]
            return And(tuple(conjuncts)) if conjuncts else BoolLit(False)

        if isinstance(ty2, UnionType):
            # ty1 must be a subtype of at least one alternative in ty2
            disjuncts = [
                self.encode_subtype_check(ty1, alt, context)
                for alt in ty2.alternatives
            ]
            # This is trickier for the negation encoding...
            # For now, check each separately
            return Or(tuple(disjuncts)) if disjuncts else BoolLit(True)

        return BoolLit(True)

    def _encode_function_subtype(
        self,
        ty1: FunctionType,
        ty2: FunctionType,
        context: TypingContext,
    ) -> Predicate:
        """Function subtyping: contravariant params, covariant return."""
        if len(ty1.param_types) != len(ty2.param_types):
            return BoolLit(True)  # incompatible

        parts: List[Predicate] = []
        # Contravariant parameters: ty2.param <: ty1.param
        for pt1, pt2 in zip(ty1.param_types, ty2.param_types):
            parts.append(self.encode_subtype_check(pt2, pt1, context))

        # Covariant return: ty1.return <: ty2.return
        parts.append(self.encode_subtype_check(ty1.return_type, ty2.return_type, context))

        if not parts:
            return BoolLit(False)
        return And(tuple(parts))

    def encode_refinement_validity(self, refined: RefinedType) -> Predicate:
        """Check if a refinement is satisfiable (non-vacuous)."""
        return refined.refinement if isinstance(refined, RefinedType) else BoolLit(True)

    def encode_path_feasibility(self, path: CfgPath) -> Predicate:
        """Encode a CFG path as an SMT formula to check feasibility."""
        constraints: List[Predicate] = []
        var_versions: Dict[str, int] = defaultdict(int)

        for edge in path.edges:
            if edge.condition is not None:
                versioned = self._version_predicate(edge.condition, var_versions)
                constraints.append(versioned)

            if edge.assignment is not None:
                var, expr = edge.assignment
                old_version = var_versions[var]
                var_versions[var] = old_version + 1
                new_var_name = f"{var}_{var_versions[var]}"
                old_expr = self._version_expr(expr, var_versions)
                constraints.append(
                    Comparison(ComparisonOp.EQ, Var(new_var_name), old_expr)
                )

        if not constraints:
            return BoolLit(True)
        return And(tuple(constraints))

    def encode_counterexample(self, ce: CfgPath) -> Predicate:
        """Encode a counterexample path as SMT formula."""
        return self.encode_path_feasibility(ce)

    def encode_predicate(self, pred: Predicate) -> Predicate:
        """Identity encoding (predicates are already in our AST)."""
        return pred

    def encode_type(self, ty: RefinementType) -> str:
        """Return the SMT sort name for a type."""
        if isinstance(ty, BaseType):
            tag_to_sort = {
                "int": "Int", "float": "Int", "bool": "Bool",
                "str": "String", "NoneType": "Int",
            }
            return tag_to_sort.get(ty.tag, "Int")
        if isinstance(ty, RefinedType):
            return self.encode_type(ty.base)
        return "Int"

    def _version_predicate(
        self, pred: Predicate, versions: Dict[str, int]
    ) -> Predicate:
        """Version variables in a predicate for SSA-like encoding."""
        if isinstance(pred, Comparison):
            return Comparison(
                pred.op,
                self._version_expr(pred.left, versions),
                self._version_expr(pred.right, versions),
            )
        if isinstance(pred, IsInstance):
            v = versions.get(pred.var, 0)
            return IsInstance(f"{pred.var}_{v}" if v > 0 else pred.var, pred.tag)
        if isinstance(pred, IsNone):
            v = versions.get(pred.var, 0)
            return IsNone(f"{pred.var}_{v}" if v > 0 else pred.var)
        if isinstance(pred, IsTruthy):
            v = versions.get(pred.var, 0)
            return IsTruthy(f"{pred.var}_{v}" if v > 0 else pred.var)
        if isinstance(pred, HasAttr):
            v = versions.get(pred.var, 0)
            return HasAttr(f"{pred.var}_{v}" if v > 0 else pred.var, pred.key)
        if isinstance(pred, And):
            return And(tuple(self._version_predicate(c, versions) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return Or(tuple(self._version_predicate(d, versions) for d in pred.disjuncts))
        if isinstance(pred, Not):
            return Not(self._version_predicate(pred.operand, versions))
        if isinstance(pred, Implies):
            return Implies(
                self._version_predicate(pred.antecedent, versions),
                self._version_predicate(pred.consequent, versions),
            )
        return pred

    def _version_expr(self, expr: Expr, versions: Dict[str, int]) -> Expr:
        if isinstance(expr, Var):
            v = versions.get(expr.name, 0)
            return Var(f"{expr.name}_{v}" if v > 0 else expr.name, expr.sort)
        if isinstance(expr, Const):
            return expr
        if isinstance(expr, Len):
            return Len(self._version_expr(expr.arg, versions))
        if isinstance(expr, BinOp):
            return BinOp(
                expr.op,
                self._version_expr(expr.left, versions),
                self._version_expr(expr.right, versions),
            )
        if isinstance(expr, UnaryOp):
            return UnaryOp(expr.op, self._version_expr(expr.operand, versions))
        return expr

    @staticmethod
    def _base_subtype(b1: BaseType, b2: BaseType) -> bool:
        """Check if base type b1 is a subtype of b2."""
        if b1.tag == b2.tag:
            return True
        hierarchy = {
            ("bool", "int"): True,
            ("int", "float"): True,
            ("bool", "float"): True,
            ("int", "complex"): True,
            ("float", "complex"): True,
        }
        return hierarchy.get((b1.tag, b2.tag), False)

    @staticmethod
    def _context_to_pred(ctx: TypingContext) -> Predicate:
        """Convert typing context assumptions to a predicate."""
        parts: List[Predicate] = list(ctx.assumptions)
        for name, ty in ctx.bindings.items():
            if isinstance(ty, RefinedType):
                parts.append(ty.refinement)
            elif isinstance(ty, BaseType):
                parts.append(IsInstance(name, ty.tag))
        if not parts:
            return BoolLit(True)
        if len(parts) == 1:
            return parts[0]
        return And(tuple(parts))

    @staticmethod
    def _rename_in_pred(pred: Predicate, old: str, new: str) -> Predicate:
        """Rename a variable in a predicate."""
        if isinstance(pred, Comparison):
            return Comparison(
                pred.op,
                SmtEncoder._rename_in_expr(pred.left, old, new),
                SmtEncoder._rename_in_expr(pred.right, old, new),
            )
        if isinstance(pred, IsInstance):
            return IsInstance(new if pred.var == old else pred.var, pred.tag)
        if isinstance(pred, IsNone):
            return IsNone(new if pred.var == old else pred.var)
        if isinstance(pred, IsTruthy):
            return IsTruthy(new if pred.var == old else pred.var)
        if isinstance(pred, HasAttr):
            return HasAttr(new if pred.var == old else pred.var, pred.key)
        if isinstance(pred, And):
            return And(tuple(SmtEncoder._rename_in_pred(c, old, new) for c in pred.conjuncts))
        if isinstance(pred, Or):
            return Or(tuple(SmtEncoder._rename_in_pred(d, old, new) for d in pred.disjuncts))
        if isinstance(pred, Not):
            return Not(SmtEncoder._rename_in_pred(pred.operand, old, new))
        if isinstance(pred, Implies):
            return Implies(
                SmtEncoder._rename_in_pred(pred.antecedent, old, new),
                SmtEncoder._rename_in_pred(pred.consequent, old, new),
            )
        if isinstance(pred, Iff):
            return Iff(
                SmtEncoder._rename_in_pred(pred.left, old, new),
                SmtEncoder._rename_in_pred(pred.right, old, new),
            )
        return pred

    @staticmethod
    def _rename_in_expr(expr: Expr, old: str, new: str) -> Expr:
        if isinstance(expr, Var):
            return Var(new if expr.name == old else expr.name, expr.sort)
        if isinstance(expr, Const):
            return expr
        if isinstance(expr, Len):
            return Len(SmtEncoder._rename_in_expr(expr.arg, old, new))
        if isinstance(expr, BinOp):
            return BinOp(
                expr.op,
                SmtEncoder._rename_in_expr(expr.left, old, new),
                SmtEncoder._rename_in_expr(expr.right, old, new),
            )
        if isinstance(expr, UnaryOp):
            return UnaryOp(
                expr.op,
                SmtEncoder._rename_in_expr(expr.operand, old, new),
            )
        return expr


# ===================================================================
# SubtypeChecker
# ===================================================================

class SubtypeChecker:
    """Check refinement subtyping via SMT."""

    def __init__(self, solver: Optional[SmtSolver] = None) -> None:
        self._solver = solver or FallbackSolver()
        self._encoder = SmtEncoder()
        self._cache: Dict[str, bool] = {}

    def check_subtype(
        self,
        context: TypingContext,
        ty1: RefinementType,
        ty2: RefinementType,
    ) -> bool:
        """Check if ty1 <: ty2 under context Γ.

        For refinement types {x:τ₁|φ₁} <: {x:τ₂|φ₂}:
          Γ, φ₁ ⊨ φ₂
        """
        cache_key = self._make_cache_key(context, ty1, ty2)
        if cache_key in self._cache:
            return self._cache[cache_key]

        formula = self._encoder.encode_subtype_check(ty1, ty2, context)
        # The formula is the negation: context ∧ φ₁ ∧ ¬φ₂
        # Subtype holds iff this is UNSAT

        self._solver.push()
        self._solver.assert_formula(formula)
        result = self._solver.check_sat()
        self._solver.pop()

        is_subtype = (result == SatResult.UNSAT)
        self._cache[cache_key] = is_subtype
        return is_subtype

    def check_structural_subtype(
        self,
        ty1: RefinementType,
        ty2: RefinementType,
    ) -> bool:
        """Structural subtyping with width subtyping (for records/objects)."""
        if isinstance(ty1, BaseType) and isinstance(ty2, BaseType):
            return self._encoder._base_subtype(ty1, ty2)
        if isinstance(ty1, RefinedType) and isinstance(ty2, RefinedType):
            if not self._encoder._base_subtype(ty1.base, ty2.base):
                return False
            return self.check_subtype(TypingContext(), ty1, ty2)
        if isinstance(ty1, FunctionType) and isinstance(ty2, FunctionType):
            return self._check_function_subtype(ty1, ty2)
        if isinstance(ty1, UnionType):
            return all(
                self.check_structural_subtype(alt, ty2)
                for alt in ty1.alternatives
            )
        if isinstance(ty2, UnionType):
            return any(
                self.check_structural_subtype(ty1, alt)
                for alt in ty2.alternatives
            )
        return False

    def check_recursive_subtype(
        self,
        ty1: RecursiveType,
        ty2: RecursiveType,
        *,
        assumptions: Optional[Set[Tuple[str, str]]] = None,
    ) -> bool:
        """Check recursive type subtyping with coinduction."""
        if assumptions is None:
            assumptions = set()

        pair = (ty1.name, ty2.name)
        if pair in assumptions:
            return True  # coinductive assumption

        new_assumptions = assumptions | {pair}
        unfolded1 = self._unfold_recursive(ty1)
        unfolded2 = self._unfold_recursive(ty2)

        if isinstance(unfolded1, RecursiveType) and isinstance(unfolded2, RecursiveType):
            return self.check_recursive_subtype(
                unfolded1, unfolded2, assumptions=new_assumptions
            )
        return self.check_structural_subtype(unfolded1, unfolded2)

    def _check_function_subtype(
        self, ty1: FunctionType, ty2: FunctionType
    ) -> bool:
        """Function subtyping: contravariant params, covariant return."""
        if len(ty1.param_types) != len(ty2.param_types):
            return False
        # Contravariant params: ty2.param <: ty1.param
        for pt1, pt2 in zip(ty1.param_types, ty2.param_types):
            if not self.check_structural_subtype(pt2, pt1):
                return False
        # Covariant return
        return self.check_structural_subtype(ty1.return_type, ty2.return_type)

    @staticmethod
    def _unfold_recursive(ty: RecursiveType) -> RefinementType:
        """Unfold one level of a recursive type."""
        return ty.body

    @staticmethod
    def _make_cache_key(
        ctx: TypingContext, ty1: RefinementType, ty2: RefinementType
    ) -> str:
        return f"{repr(ctx.bindings)}|{repr(ctx.assumptions)}|{repr(ty1)}|{repr(ty2)}"


# ===================================================================
# CounterexampleChecker
# ===================================================================

@dataclass
class FeasibilityResult:
    feasible: bool
    model: Optional[SmtModel] = None
    unsat_core: Optional[SmtUnsatCore] = None


class CounterexampleChecker:
    """Check feasibility of counterexample paths."""

    def __init__(self, solver: Optional[SmtSolver] = None) -> None:
        self._solver = solver or FallbackSolver()
        self._encoder = SmtEncoder()

    def check_feasibility(self, path: CfgPath) -> FeasibilityResult:
        """Check if a counterexample path is feasible.

        Returns (feasible, model_or_core).
        """
        formula = self._encoder.encode_path_feasibility(path)

        self._solver.push()
        self._solver.assert_formula(formula)
        result = self._solver.check_sat()

        if result == SatResult.SAT:
            model = self._solver.get_model()
            self._solver.pop()
            return FeasibilityResult(feasible=True, model=model)
        elif result == SatResult.UNSAT:
            core = self._solver.get_unsat_core()
            self._solver.pop()
            return FeasibilityResult(feasible=False, unsat_core=core)
        else:
            self._solver.pop()
            return FeasibilityResult(feasible=True)  # conservative

    def extract_concrete_witness(
        self, model: SmtModel, path: CfgPath
    ) -> Dict[str, Any]:
        """Extract a concrete witness (input values) from a model."""
        # The model contains versioned variables; extract the initial versions
        witness: Dict[str, Any] = {}
        for name, value in model.variable_values.items():
            # Remove version suffix to get original name
            base = name.rsplit("_", 1)[0] if "_" in name else name
            if base not in witness:
                witness[base] = value
        return witness

    def check_path_batch(
        self, paths: List[CfgPath]
    ) -> List[FeasibilityResult]:
        """Check feasibility of multiple paths."""
        return [self.check_feasibility(path) for path in paths]


# ===================================================================
# InterpolantExtractor
# ===================================================================

class InterpolantExtractor:
    """Unsat core-based predicate extraction for contract discovery."""

    def __init__(self, solver: Optional[SmtSolver] = None) -> None:
        self._solver = solver or FallbackSolver()
        self._encoder = SmtEncoder()

    def extract_interpolant(
        self, a: Predicate, b: Predicate
    ) -> Optional[Predicate]:
        """Extract predicates from the unsat core of A ∧ B.

        Given A ∧ B is unsat, extracts predicates from the
        unsat core to discover relevant constraints.
        """
        if isinstance(self._solver, Z3Solver):
            return self._solver.compute_interpolant(a, b)

        # Fallback: syntactic predicate extraction
        return self._syntactic_interpolation(a, b)

    def sequence_interpolation(
        self, formulas: List[Predicate]
    ) -> List[Optional[Predicate]]:
        """Compute sequence interpolants for a path.

        Given A₀, A₁, ..., Aₙ with A₀ ∧ A₁ ∧ ... ∧ Aₙ is unsat,
        find I₁, ..., Iₙ₋₁ such that:
          A₀ ⊨ I₁
          Iᵢ ∧ Aᵢ ⊨ Iᵢ₊₁
          Iₙ₋₁ ∧ Aₙ ⊨ ⊥
        """
        if len(formulas) < 2:
            return []

        interpolants: List[Optional[Predicate]] = []
        prefix = formulas[0]

        for i in range(1, len(formulas)):
            suffix_parts = formulas[i:]
            suffix = And(tuple(suffix_parts)) if len(suffix_parts) > 1 else suffix_parts[0]

            interp = self.extract_interpolant(prefix, suffix)
            interpolants.append(interp)

            if interp is not None:
                prefix = And((prefix, formulas[i]))
            else:
                prefix = And((prefix, formulas[i]))

        return interpolants

    def tree_interpolation(
        self,
        nodes: List[Predicate],
        children: Dict[int, List[int]],
    ) -> Dict[int, Optional[Predicate]]:
        """Compute tree interpolants for branching paths.

        *nodes* is a list of formulas at each tree node.
        *children[i]* gives the children of node i.
        """
        result: Dict[int, Optional[Predicate]] = {}
        # Process leaves first, then parents
        processed: Set[int] = set()

        def process(node: int) -> Optional[Predicate]:
            if node in processed:
                return result.get(node)
            processed.add(node)

            child_ids = children.get(node, [])
            if not child_ids:
                # Leaf: interpolate against rest
                result[node] = nodes[node] if node < len(nodes) else BoolLit(True)
                return result[node]

            # Process children first
            child_formulas = []
            for c in child_ids:
                cf = process(c)
                if cf is not None:
                    child_formulas.append(cf)

            # Node formula with child interpolants
            node_formula = nodes[node] if node < len(nodes) else BoolLit(True)
            if child_formulas:
                combined = And(tuple([node_formula] + child_formulas))
            else:
                combined = node_formula

            result[node] = combined
            return combined

        # Process all roots (nodes with no parent)
        all_children: Set[int] = set()
        for cs in children.values():
            all_children.update(cs)
        roots = [i for i in range(len(nodes)) if i not in all_children]

        for root in roots:
            process(root)

        return result

    def project_to_predicate_language(
        self, interpolant: Predicate
    ) -> Predicate:
        """Project an interpolant into the predicate language P."""
        return self._simplify_to_P(interpolant)

    def _syntactic_interpolation(
        self, a: Predicate, b: Predicate
    ) -> Optional[Predicate]:
        """Syntactic predicate extraction fallback."""
        common_vars = a.free_vars() & b.free_vars()
        if not common_vars:
            return BoolLit(False)

        # Extract atoms from A that only use common variables
        a_atoms = self._extract_relevant_atoms(a, common_vars)
        if a_atoms:
            return And(tuple(a_atoms)) if len(a_atoms) > 1 else a_atoms[0]
        return None

    def _extract_relevant_atoms(
        self, pred: Predicate, target_vars: FrozenSet[str]
    ) -> List[Predicate]:
        """Extract atomic predicates whose free vars are in target_vars."""
        result: List[Predicate] = []
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
            if pred.free_vars() <= target_vars:
                result.append(pred)
        elif isinstance(pred, Not):
            if isinstance(pred.operand, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
                if pred.operand.free_vars() <= target_vars:
                    result.append(pred)
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                result.extend(self._extract_relevant_atoms(c, target_vars))
        elif isinstance(pred, Or):
            for d in pred.disjuncts:
                result.extend(self._extract_relevant_atoms(d, target_vars))
        return result

    @staticmethod
    def _simplify_to_P(pred: Predicate) -> Predicate:
        """Ensure predicate uses only the language P constructs."""
        return pred


# ===================================================================
# ModelExtractor
# ===================================================================

class ModelExtractor:
    """Extract information from SMT models."""

    def get_variable_values(self, model: SmtModel) -> Dict[str, Any]:
        """Get all variable values from the model."""
        return dict(model.variable_values)

    def get_function_interpretations(self, model: SmtModel) -> Dict[str, Any]:
        """Get all function interpretations from the model."""
        return dict(model.function_interpretations)

    def reconstruct_concrete_state(
        self, model: SmtModel
    ) -> Dict[str, Any]:
        """Reconstruct a concrete program state from the model.

        Maps versioned variables back to their base names with final values.
        """
        state: Dict[str, Any] = {}
        versioned: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)

        for name, value in model.variable_values.items():
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base = parts[0]
                version = int(parts[1])
                versioned[base].append((version, value))
            else:
                state[name] = value

        # Take the highest version for each variable
        for base, versions in versioned.items():
            versions.sort(key=lambda x: x[0])
            state[base] = versions[-1][1]

        return state

    def get_counterexample_path(
        self, model: SmtModel, path: CfgPath
    ) -> List[Dict[str, Any]]:
        """Reconstruct the counterexample path with concrete values at each step."""
        trace: List[Dict[str, Any]] = []
        current_state = dict(path.entry_state)

        for edge in path.edges:
            step: Dict[str, Any] = {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "state": dict(current_state),
            }
            if edge.condition is not None:
                step["condition"] = repr(edge.condition)
            if edge.assignment is not None:
                var, expr = edge.assignment
                step["assignment"] = f"{var} = ..."
                value = model.variable_values.get(var)
                if value is not None:
                    current_state[var] = value
            trace.append(step)

        return trace

    def extract_type_witnesses(
        self, model: SmtModel
    ) -> Dict[str, str]:
        """Extract type tag witnesses from the model."""
        types: Dict[str, str] = {}
        for name, value in model.variable_values.items():
            if name.startswith("typeof_"):
                var = name[len("typeof_"):]
                types[var] = str(value)
        return types

    def format_model(self, model: SmtModel) -> str:
        """Format a model for human-readable display."""
        lines: List[str] = []
        for name, value in sorted(model.variable_values.items()):
            lines.append(f"  {name} = {value!r}")
        if model.function_interpretations:
            lines.append("  Functions:")
            for name, interp in sorted(model.function_interpretations.items()):
                lines.append(f"    {name}: {interp}")
        return "\n".join(lines)


# ===================================================================
# TypeTagEncoder
# ===================================================================

class TypeTagEncoder:
    """Encode type tag reasoning for SMT."""

    # Hierarchy: parent → set of child tags
    TAG_HIERARCHY: Dict[str, Set[str]] = {
        "Number": {"int", "float", "complex", "bool"},
        "Integral": {"int", "bool"},
        "Real": {"int", "float", "bool"},
        "Sequence": {"list", "tuple", "str", "bytes", "bytearray", "range"},
        "Mapping": {"dict"},
        "Set": {"set", "frozenset"},
        "Iterable": {"list", "tuple", "str", "bytes", "bytearray", "range",
                     "dict", "set", "frozenset"},
        "Sized": {"list", "tuple", "str", "bytes", "bytearray", "range",
                  "dict", "set", "frozenset"},
        "object": {"int", "float", "complex", "bool", "str", "bytes",
                   "bytearray", "list", "tuple", "dict", "set", "frozenset",
                   "NoneType", "range", "type", "memoryview"},
    }

    def __init__(self) -> None:
        self._all_tags: List[str] = [
            "int", "float", "str", "bool", "list", "tuple", "dict", "set",
            "bytes", "NoneType", "complex", "frozenset", "bytearray",
            "memoryview", "range", "type",
        ]

    def encode_isinstance_test(
        self, var: str, tag: str
    ) -> Predicate:
        """Encode isinstance(var, tag) with hierarchy."""
        children = self.TAG_HIERARCHY.get(tag)
        if children is not None:
            # Abstract type: isinstance(x, Number) ↔ isinstance(x, int) ∨ ...
            return Or(tuple(IsInstance(var, child) for child in sorted(children)))
        return IsInstance(var, tag)

    def encode_tag_hierarchy(
        self, var: str
    ) -> List[Predicate]:
        """Generate axioms for the type tag hierarchy."""
        axioms: List[Predicate] = []
        for parent, children in self.TAG_HIERARCHY.items():
            for child in children:
                # isinstance(x, child) → isinstance(x, parent)
                axioms.append(
                    Implies(IsInstance(var, child), IsInstance(var, parent))
                )
        return axioms

    def encode_tag_exclusivity(
        self, var: str
    ) -> List[Predicate]:
        """Generate axioms: exactly one concrete tag holds."""
        concrete_tags = self._all_tags
        # At least one
        at_least_one = Or(tuple(IsInstance(var, tag) for tag in concrete_tags))
        axioms: List[Predicate] = [at_least_one]

        # Pairwise exclusivity
        for i, t1 in enumerate(concrete_tags):
            for t2 in concrete_tags[i + 1:]:
                axioms.append(
                    Not(And((IsInstance(var, t1), IsInstance(var, t2))))
                )
        return axioms

    def encode_none_type(self, var: str) -> List[Predicate]:
        """Encode None type constraints."""
        return [
            # isinstance(x, NoneType) ↔ is_none(x)
            Iff(IsInstance(var, "NoneType"), IsNone(var)),
            # is_none(x) → ¬is_truthy(x)
            Implies(IsNone(var), Not(IsTruthy(var))),
        ]

    def is_subtype_tag(self, sub: str, super_tag: str) -> bool:
        """Check if sub is a subtype tag of super_tag."""
        if sub == super_tag:
            return True
        children = self.TAG_HIERARCHY.get(super_tag)
        if children is not None:
            return sub in children
        return False

    def get_all_supertypes(self, tag: str) -> Set[str]:
        """Get all supertypes of a given tag."""
        result: Set[str] = {tag}
        for parent, children in self.TAG_HIERARCHY.items():
            if tag in children:
                result.add(parent)
        return result

    def get_all_subtypes(self, tag: str) -> Set[str]:
        """Get all subtypes of a given tag."""
        children = self.TAG_HIERARCHY.get(tag)
        if children is not None:
            return set(children)
        return {tag}


# ===================================================================
# ArithmeticEncoder
# ===================================================================

class ArithmeticEncoder:
    """Encode arithmetic reasoning for SMT."""

    def encode_linear_arithmetic(
        self, expr: Expr
    ) -> Predicate:
        """Encode a linear arithmetic expression as constraints."""
        # Just pass through — expressions are already in our AST
        return BoolLit(True)

    def encode_division(
        self, dividend: Expr, divisor: Expr, result_var: str
    ) -> List[Predicate]:
        """Encode integer division with preconditions.

        q = dividend / divisor  (integer division)
        Encodes: divisor != 0 ∧ dividend = q * divisor + r ∧ 0 <= r < |divisor|
        """
        q = Var(result_var)
        r_var = f"{result_var}_rem"
        r = Var(r_var)
        constraints: List[Predicate] = [
            # divisor != 0
            Comparison(ComparisonOp.NE, divisor, Const(0)),
            # dividend = q * divisor + r
            Comparison(
                ComparisonOp.EQ,
                dividend,
                BinOp(ArithOp.ADD, BinOp(ArithOp.MUL, q, divisor), r),
            ),
            # 0 <= r
            Comparison(ComparisonOp.GE, r, Const(0)),
            # r < |divisor|
            Comparison(
                ComparisonOp.LT,
                r,
                UnaryOp(UnaryArithOp.ABS, divisor),
            ),
        ]
        return constraints

    def encode_modulo(
        self, dividend: Expr, divisor: Expr, result_var: str
    ) -> List[Predicate]:
        """Encode modulo operation with preconditions."""
        r = Var(result_var)
        q_var = f"{result_var}_quot"
        q = Var(q_var)
        constraints: List[Predicate] = [
            Comparison(ComparisonOp.NE, divisor, Const(0)),
            Comparison(
                ComparisonOp.EQ,
                dividend,
                BinOp(ArithOp.ADD, BinOp(ArithOp.MUL, q, divisor), r),
            ),
            Comparison(ComparisonOp.GE, r, Const(0)),
            Comparison(ComparisonOp.LT, r, UnaryOp(UnaryArithOp.ABS, divisor)),
        ]
        return constraints

    def encode_length(self, var: str, len_var: str) -> List[Predicate]:
        """Encode length function constraints."""
        return [
            # len(x) >= 0
            Comparison(ComparisonOp.GE, Var(len_var), Const(0)),
            # len(x) = len_fn(x)
            Comparison(ComparisonOp.EQ, Var(len_var), Len(Var(var))),
        ]

    def encode_array_bounds(
        self, array_var: str, index_var: str, len_var: str
    ) -> List[Predicate]:
        """Encode array bounds checking: 0 <= index < len(array)."""
        return [
            Comparison(ComparisonOp.GE, Var(index_var), Const(0)),
            Comparison(ComparisonOp.LT, Var(index_var), Var(len_var)),
        ]

    def encode_non_negative(self, var: str) -> Predicate:
        """Encode var >= 0."""
        return Comparison(ComparisonOp.GE, Var(var), Const(0))

    def encode_abs(self, var: str, abs_var: str) -> List[Predicate]:
        """Encode absolute value: abs_var = |var|."""
        return [
            Comparison(ComparisonOp.GE, Var(abs_var), Const(0)),
            Or((
                And((
                    Comparison(ComparisonOp.GE, Var(var), Const(0)),
                    Comparison(ComparisonOp.EQ, Var(abs_var), Var(var)),
                )),
                And((
                    Comparison(ComparisonOp.LT, Var(var), Const(0)),
                    Comparison(
                        ComparisonOp.EQ,
                        Var(abs_var),
                        UnaryOp(UnaryArithOp.NEG, Var(var)),
                    ),
                )),
            )),
        ]


# ===================================================================
# ContractVerifier
# ===================================================================

@dataclass
class FunctionContract:
    """A function contract (precondition, postcondition)."""
    function_name: str
    param_names: List[str]
    precondition: Predicate
    postcondition: Predicate
    return_var: str = "_ret"


@dataclass
class LoopInvariant:
    """A loop invariant."""
    loop_id: str
    invariant: Predicate
    loop_var: str = "_i"


@dataclass
class VerificationResult:
    """Result of a verification check."""
    verified: bool
    counterexample: Optional[SmtModel] = None
    message: str = ""


class ContractVerifier:
    """Verify inferred contracts using SMT."""

    def __init__(self, solver: Optional[SmtSolver] = None) -> None:
        self._solver = solver or FallbackSolver()
        self._encoder = SmtEncoder()

    def verify_function_contract(
        self, contract: FunctionContract
    ) -> VerificationResult:
        """Verify a function contract: pre ⊨ post.

        Checks that the precondition implies the postcondition.
        """
        self._solver.push()

        # Assert precondition
        self._solver.assert_formula(contract.precondition)
        # Assert negation of postcondition
        self._solver.assert_formula(Not(contract.postcondition))

        result = self._solver.check_sat()
        self._solver.pop()

        if result == SatResult.UNSAT:
            return VerificationResult(
                verified=True,
                message=f"Contract for {contract.function_name} verified.",
            )
        elif result == SatResult.SAT:
            model = self._solver.get_model()
            return VerificationResult(
                verified=False,
                counterexample=model,
                message=f"Contract for {contract.function_name} violated.",
            )
        else:
            return VerificationResult(
                verified=False,
                message=f"Contract verification for {contract.function_name} inconclusive.",
            )

    def verify_loop_invariant(
        self, invariant: LoopInvariant, loop_body: Predicate
    ) -> VerificationResult:
        """Verify a loop invariant: inv ∧ body ⊨ inv'.

        Checks that the invariant is preserved by the loop body.
        """
        self._solver.push()

        # Assert invariant and loop body
        self._solver.assert_formula(invariant.invariant)
        self._solver.assert_formula(loop_body)

        # Assert negation of invariant (for the next iteration)
        primed_inv = self._prime_predicate(invariant.invariant)
        self._solver.assert_formula(Not(primed_inv))

        result = self._solver.check_sat()
        self._solver.pop()

        if result == SatResult.UNSAT:
            return VerificationResult(
                verified=True,
                message=f"Loop invariant for {invariant.loop_id} verified.",
            )
        elif result == SatResult.SAT:
            model = self._solver.get_model()
            return VerificationResult(
                verified=False,
                counterexample=model,
                message=f"Loop invariant for {invariant.loop_id} violated.",
            )
        else:
            return VerificationResult(
                verified=False,
                message=f"Loop invariant verification for {invariant.loop_id} inconclusive.",
            )

    def generate_proof_certificate(
        self, verification: VerificationResult, contract: FunctionContract
    ) -> str:
        """Generate an SMT-LIB proof certificate."""
        lines: List[str] = []
        lines.append(f"; Proof certificate for {contract.function_name}")
        lines.append(f"; Verified: {verification.verified}")
        lines.append(f"(set-logic QF_UFLIA)")

        # Declare variables
        for param in contract.param_names:
            lines.append(f"(declare-const {param} Int)")
        lines.append(f"(declare-const {contract.return_var} Int)")

        # Assert precondition
        lines.append(f"; Precondition")
        lines.append(f"(assert {self._pred_to_smt(contract.precondition)})")

        # Assert negation of postcondition
        lines.append(f"; Negation of postcondition")
        lines.append(
            f"(assert (not {self._pred_to_smt(contract.postcondition)}))"
        )

        lines.append("(check-sat)")
        lines.append(f"; Expected: unsat (if contract is valid)")

        return "\n".join(lines)

    def _prime_predicate(self, pred: Predicate) -> Predicate:
        """Create a 'primed' version of a predicate (x → x')."""
        return SmtEncoder._rename_in_pred(pred, pred.free_vars().__iter__().__next__(), 
                                           pred.free_vars().__iter__().__next__() + "'") if pred.free_vars() else pred

    def _pred_to_smt(self, pred: Predicate) -> str:
        """Convert predicate to SMT-LIB string."""
        if isinstance(pred, BoolLit):
            return "true" if pred.value else "false"
        if isinstance(pred, Comparison):
            op_map = {
                ComparisonOp.EQ: "=", ComparisonOp.NE: "distinct",
                ComparisonOp.LT: "<", ComparisonOp.LE: "<=",
                ComparisonOp.GT: ">", ComparisonOp.GE: ">=",
            }
            return f"({op_map[pred.op]} {self._expr_to_smt(pred.left)} {self._expr_to_smt(pred.right)})"
        if isinstance(pred, IsInstance):
            return f"(= (typeof {pred.var}) {pred.tag})"
        if isinstance(pred, IsNone):
            return f"(is_none {pred.var})"
        if isinstance(pred, IsTruthy):
            return f"(is_truthy {pred.var})"
        if isinstance(pred, HasAttr):
            return f"(hasattr {pred.var} \"{pred.key}\")"
        if isinstance(pred, Not):
            return f"(not {self._pred_to_smt(pred.operand)})"
        if isinstance(pred, And):
            parts = " ".join(self._pred_to_smt(c) for c in pred.conjuncts)
            return f"(and {parts})"
        if isinstance(pred, Or):
            parts = " ".join(self._pred_to_smt(d) for d in pred.disjuncts)
            return f"(or {parts})"
        if isinstance(pred, Implies):
            return f"(=> {self._pred_to_smt(pred.antecedent)} {self._pred_to_smt(pred.consequent)})"
        return "true"

    def _expr_to_smt(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            if isinstance(expr.value, int):
                return str(expr.value) if expr.value >= 0 else f"(- {abs(expr.value)})"
            return repr(expr.value)
        if isinstance(expr, Len):
            return f"(len {self._expr_to_smt(expr.arg)})"
        if isinstance(expr, BinOp):
            op_map = {
                ArithOp.ADD: "+", ArithOp.SUB: "-", ArithOp.MUL: "*",
                ArithOp.DIV: "div", ArithOp.MOD: "mod",
            }
            return f"({op_map[expr.op]} {self._expr_to_smt(expr.left)} {self._expr_to_smt(expr.right)})"
        if isinstance(expr, UnaryOp):
            if expr.op == UnaryArithOp.NEG:
                return f"(- {self._expr_to_smt(expr.operand)})"
            if expr.op == UnaryArithOp.ABS:
                return f"(abs {self._expr_to_smt(expr.operand)})"
        return "0"


# ===================================================================
# SmtCache
# ===================================================================

class SmtCache:
    """Cache SMT query results for performance."""

    def __init__(self, *, max_size: int = 10000) -> None:
        self._cache: Dict[str, SmtQueryResult] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._total_time_saved = 0.0

    def normalize_key(self, formula: Predicate) -> str:
        """Normalize a formula for use as a cache key."""
        return hashlib.sha256(repr(formula).encode()).hexdigest()

    def lookup(self, formula: Predicate) -> Optional[SmtQueryResult]:
        """Look up a cached result."""
        key = self.normalize_key(formula)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            self._total_time_saved += result.time_seconds
            return result
        self._misses += 1
        return None

    def store(self, formula: Predicate, result: SmtQueryResult) -> None:
        """Store a result in the cache."""
        if len(self._cache) >= self._max_size:
            self._evict()
        key = self.normalize_key(formula)
        self._cache[key] = result

    def invalidate(self, formula: Predicate) -> None:
        """Invalidate a cached result."""
        key = self.normalize_key(formula)
        self._cache.pop(key, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    def _evict(self) -> None:
        """Evict entries to make room (LRU-like: remove oldest)."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def statistics(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "time_saved": self._total_time_saved,
        }


# ===================================================================
# SmtStatistics
# ===================================================================

class SmtStatistics:
    """Track solver statistics."""

    def __init__(self) -> None:
        self._query_counts: Dict[str, int] = defaultdict(int)
        self._solve_times: List[float] = []
        self._result_counts: Dict[SatResult, int] = defaultdict(int)
        self._cache_stats: Dict[str, int] = defaultdict(int)
        self._total_queries = 0
        self._start_time = time.monotonic()

    def record_query(
        self,
        query_type: str,
        result: SatResult,
        time_seconds: float,
    ) -> None:
        """Record a query for statistics."""
        self._query_counts[query_type] += 1
        self._solve_times.append(time_seconds)
        self._result_counts[result] += 1
        self._total_queries += 1

    def record_cache_hit(self) -> None:
        self._cache_stats["hits"] += 1

    def record_cache_miss(self) -> None:
        self._cache_stats["misses"] += 1

    @property
    def total_queries(self) -> int:
        return self._total_queries

    @property
    def total_time(self) -> float:
        return sum(self._solve_times)

    @property
    def average_time(self) -> float:
        return self.total_time / self._total_queries if self._total_queries > 0 else 0.0

    @property
    def median_time(self) -> float:
        if not self._solve_times:
            return 0.0
        sorted_times = sorted(self._solve_times)
        mid = len(sorted_times) // 2
        if len(sorted_times) % 2 == 0:
            return (sorted_times[mid - 1] + sorted_times[mid]) / 2
        return sorted_times[mid]

    def query_counts_by_type(self) -> Dict[str, int]:
        return dict(self._query_counts)

    def result_distribution(self) -> Dict[str, int]:
        return {r.value: c for r, c in self._result_counts.items()}

    def time_histogram(self, buckets: int = 10) -> List[Tuple[float, float, int]]:
        """Compute a histogram of solve times."""
        if not self._solve_times:
            return []
        min_t = min(self._solve_times)
        max_t = max(self._solve_times)
        if min_t == max_t:
            return [(min_t, max_t, len(self._solve_times))]

        width = (max_t - min_t) / buckets
        histogram: List[Tuple[float, float, int]] = []
        for i in range(buckets):
            lo = min_t + i * width
            hi = lo + width
            count = sum(1 for t in self._solve_times if lo <= t < hi)
            histogram.append((lo, hi, count))
        return histogram

    def cache_hit_rate(self) -> float:
        total = self._cache_stats.get("hits", 0) + self._cache_stats.get("misses", 0)
        if total == 0:
            return 0.0
        return self._cache_stats["hits"] / total

    def summary(self) -> Dict[str, Any]:
        return {
            "total_queries": self._total_queries,
            "total_time": self.total_time,
            "average_time": self.average_time,
            "median_time": self.median_time,
            "query_counts": self.query_counts_by_type(),
            "result_distribution": self.result_distribution(),
            "cache_hit_rate": self.cache_hit_rate(),
            "uptime": time.monotonic() - self._start_time,
        }

    def report(self) -> str:
        """Generate a human-readable statistics report."""
        s = self.summary()
        lines = [
            "SMT Solver Statistics",
            "=" * 40,
            f"Total queries:    {s['total_queries']}",
            f"Total time:       {s['total_time']:.3f}s",
            f"Average time:     {s['average_time']:.3f}s",
            f"Median time:      {s['median_time']:.3f}s",
            f"Cache hit rate:   {s['cache_hit_rate']:.1%}",
            "",
            "Query counts by type:",
        ]
        for qtype, count in sorted(s["query_counts"].items()):
            lines.append(f"  {qtype}: {count}")
        lines.append("")
        lines.append("Result distribution:")
        for result, count in sorted(s["result_distribution"].items()):
            lines.append(f"  {result}: {count}")
        return "\n".join(lines)


# ===================================================================
# CachedSmtSolver — wraps a solver with caching + statistics
# ===================================================================

class CachedSmtSolver:
    """Wrapper that adds caching and statistics to any SmtSolver."""

    def __init__(
        self,
        solver: SmtSolver,
        *,
        cache_enabled: bool = True,
        max_cache_size: int = 10000,
    ) -> None:
        self._solver = solver
        self._cache = SmtCache(max_size=max_cache_size) if cache_enabled else None
        self._stats = SmtStatistics()

    @property
    def solver(self) -> SmtSolver:
        return self._solver

    @property
    def cache(self) -> Optional[SmtCache]:
        return self._cache

    @property
    def statistics(self) -> SmtStatistics:
        return self._stats

    def check_sat_cached(self, formula: Predicate) -> SmtQueryResult:
        """Check satisfiability with caching."""
        if self._cache is not None:
            cached = self._cache.lookup(formula)
            if cached is not None:
                self._stats.record_cache_hit()
                return cached
            self._stats.record_cache_miss()

        self._solver.push()
        self._solver.assert_formula(formula)

        start = time.monotonic()
        result = self._solver.check_sat()
        elapsed = time.monotonic() - start

        model = self._solver.get_model() if result == SatResult.SAT else None
        core = self._solver.get_unsat_core() if result == SatResult.UNSAT else None

        self._solver.pop()

        query_result = SmtQueryResult(
            result=result,
            model=model,
            unsat_core=core,
            time_seconds=elapsed,
        )

        self._stats.record_query("check_sat", result, elapsed)
        if self._cache is not None:
            self._cache.store(formula, query_result)

        return query_result

    def check_implication(
        self, premise: Predicate, conclusion: Predicate
    ) -> bool:
        """Check if premise ⊨ conclusion via SMT.

        Returns True if premise ∧ ¬conclusion is UNSAT.
        """
        negation = And((premise, Not(conclusion)))
        result = self.check_sat_cached(negation)
        return result.result == SatResult.UNSAT

    def check_equivalence(
        self, p1: Predicate, p2: Predicate
    ) -> bool:
        """Check if p1 ↔ p2."""
        return self.check_implication(p1, p2) and self.check_implication(p2, p1)

    def check_satisfiability(self, pred: Predicate) -> SmtQueryResult:
        """Check satisfiability of a predicate."""
        return self.check_sat_cached(pred)

    def find_model(self, pred: Predicate) -> Optional[SmtModel]:
        """Find a satisfying model, or None."""
        result = self.check_sat_cached(pred)
        return result.model

    def is_valid(self, pred: Predicate) -> bool:
        """Check if a predicate is valid (¬P is UNSAT)."""
        result = self.check_sat_cached(Not(pred))
        return result.result == SatResult.UNSAT

    def report(self) -> str:
        """Generate a statistics report."""
        parts = [self._stats.report()]
        if self._cache is not None:
            cache_stats = self._cache.statistics()
            parts.append(f"\nCache: {cache_stats['size']} entries, "
                        f"{cache_stats['hit_rate']:.1%} hit rate")
        return "\n".join(parts)


# ===================================================================
# Solver factory
# ===================================================================

def create_solver(
    *,
    backend: str = "auto",
    timeout_ms: int = 30000,
    cache: bool = True,
) -> CachedSmtSolver:
    """Create an SMT solver with the specified backend.

    *backend* can be ``"z3"``, ``"fallback"``, or ``"auto"`` (try Z3 first).
    """
    solver: SmtSolver
    if backend == "z3":
        solver = Z3Solver(timeout_ms=timeout_ms)
    elif backend == "fallback":
        solver = FallbackSolver(timeout_ms=timeout_ms)
    elif backend == "auto":
        try:
            solver = Z3Solver(timeout_ms=timeout_ms)
            logger.info("Using Z3 solver backend")
        except ImportError:
            solver = FallbackSolver(timeout_ms=timeout_ms)
            logger.info("Z3 not available; using fallback solver")
    else:
        raise ValueError(f"Unknown solver backend: {backend!r}")

    return CachedSmtSolver(solver, cache_enabled=cache)


# ===================================================================
# Convenience functions
# ===================================================================

def check_refinement_subtype(
    context: TypingContext,
    ty1: RefinementType,
    ty2: RefinementType,
    *,
    backend: str = "auto",
) -> bool:
    """Check if ty1 <: ty2 in context Γ."""
    cached = create_solver(backend=backend)
    checker = SubtypeChecker(cached.solver)
    return checker.check_subtype(context, ty1, ty2)


def check_path_feasibility(
    path: CfgPath,
    *,
    backend: str = "auto",
) -> FeasibilityResult:
    """Check if a CFG path is feasible."""
    cached = create_solver(backend=backend)
    checker = CounterexampleChecker(cached.solver)
    return checker.check_feasibility(path)


def verify_contract(
    contract: FunctionContract,
    *,
    backend: str = "auto",
) -> VerificationResult:
    """Verify a function contract."""
    cached = create_solver(backend=backend)
    verifier = ContractVerifier(cached.solver)
    return verifier.verify_function_contract(contract)


def extract_interpolant(
    a: Predicate,
    b: Predicate,
    *,
    backend: str = "auto",
) -> Optional[Predicate]:
    """Compute a predicate via unsat-core-based extraction."""
    cached = create_solver(backend=backend)
    extractor = InterpolantExtractor(cached.solver)
    return extractor.extract_interpolant(a, b)


# ===================================================================
# SmtBatchSolver — batch query processing
# ===================================================================

@dataclass
class BatchQuery:
    """A single query in a batch."""
    id: str
    formula: Predicate
    query_type: str = "check_sat"
    priority: int = 0


@dataclass
class BatchResult:
    """Result for a single batch query."""
    id: str
    result: SmtQueryResult


class SmtBatchSolver:
    """Process multiple SMT queries efficiently."""

    def __init__(self, solver: Optional[CachedSmtSolver] = None) -> None:
        self._solver = solver or create_solver()
        self._pending: List[BatchQuery] = []
        self._results: Dict[str, BatchResult] = {}

    def add_query(self, query: BatchQuery) -> None:
        """Add a query to the batch."""
        self._pending.append(query)

    def add_sat_check(self, id: str, formula: Predicate, *, priority: int = 0) -> None:
        """Convenience: add a satisfiability check."""
        self._pending.append(BatchQuery(id=id, formula=formula, priority=priority))

    def add_validity_check(self, id: str, formula: Predicate, *, priority: int = 0) -> None:
        """Convenience: add a validity check (is ¬formula unsat?)."""
        self._pending.append(
            BatchQuery(id=id, formula=Not(formula), query_type="validity", priority=priority)
        )

    def add_implication_check(
        self, id: str, premise: Predicate, conclusion: Predicate, *, priority: int = 0
    ) -> None:
        """Convenience: add an implication check (premise ∧ ¬conclusion unsat?)."""
        self._pending.append(
            BatchQuery(
                id=id,
                formula=And((premise, Not(conclusion))),
                query_type="implication",
                priority=priority,
            )
        )

    def process_all(self) -> Dict[str, BatchResult]:
        """Process all pending queries, returning results by id."""
        # Sort by priority (higher first)
        self._pending.sort(key=lambda q: -q.priority)

        for query in self._pending:
            qr = self._solver.check_sat_cached(query.formula)
            self._results[query.id] = BatchResult(id=query.id, result=qr)

        self._pending.clear()
        return dict(self._results)

    def get_result(self, id: str) -> Optional[BatchResult]:
        """Get the result for a specific query."""
        return self._results.get(id)

    def clear(self) -> None:
        """Clear all pending queries and results."""
        self._pending.clear()
        self._results.clear()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def result_count(self) -> int:
        return len(self._results)

    def summary(self) -> Dict[str, Any]:
        """Summary of batch processing."""
        sat_count = sum(
            1 for r in self._results.values() if r.result.result == SatResult.SAT
        )
        unsat_count = sum(
            1 for r in self._results.values() if r.result.result == SatResult.UNSAT
        )
        unknown_count = sum(
            1 for r in self._results.values()
            if r.result.result in (SatResult.UNKNOWN, SatResult.TIMEOUT)
        )
        total_time = sum(r.result.time_seconds for r in self._results.values())
        return {
            "total": len(self._results),
            "sat": sat_count,
            "unsat": unsat_count,
            "unknown": unknown_count,
            "total_time": total_time,
        }


# ===================================================================
# IncrementalSolver — incremental SMT solving for contract discovery
# ===================================================================

class IncrementalSolver:
    """Incremental SMT solver for contract discovery loops.

    Maintains a stack of assertions that can be efficiently extended
    without re-encoding the entire problem.
    """

    def __init__(self, solver: Optional[SmtSolver] = None) -> None:
        self._solver = solver or FallbackSolver()
        self._scope_depth = 0
        self._assertion_counts: List[int] = [0]
        self._stats = SmtStatistics()

    @property
    def scope_depth(self) -> int:
        return self._scope_depth

    def push_scope(self) -> None:
        """Enter a new scope."""
        self._solver.push()
        self._scope_depth += 1
        self._assertion_counts.append(0)

    def pop_scope(self) -> None:
        """Leave the current scope."""
        if self._scope_depth > 0:
            self._solver.pop()
            self._scope_depth -= 1
            self._assertion_counts.pop()

    def add_assertion(self, formula: Predicate, label: Optional[str] = None) -> None:
        """Add an assertion at the current scope."""
        self._solver.assert_formula(formula, label=label)
        self._assertion_counts[-1] += 1

    def add_predicate_refinement(self, predicates: List[Predicate]) -> None:
        """Add a set of predicate refinements (for contract discovery)."""
        for pred in predicates:
            self.add_assertion(pred)

    def check(self) -> SatResult:
        """Check satisfiability."""
        start = time.monotonic()
        result = self._solver.check_sat()
        elapsed = time.monotonic() - start
        self._stats.record_query("incremental_check", result, elapsed)
        return result

    def check_with_assumptions(self, assumptions: List[Predicate]) -> SatResult:
        """Check satisfiability with temporary assumptions."""
        start = time.monotonic()
        result = self._solver.check_sat_assuming(assumptions)
        elapsed = time.monotonic() - start
        self._stats.record_query("incremental_check_assuming", result, elapsed)
        return result

    def get_model(self) -> Optional[SmtModel]:
        return self._solver.get_model()

    def get_unsat_core(self) -> Optional[SmtUnsatCore]:
        return self._solver.get_unsat_core()

    def reset(self) -> None:
        """Full reset."""
        self._solver.reset()
        self._scope_depth = 0
        self._assertion_counts = [0]

    @property
    def assertion_count(self) -> int:
        return sum(self._assertion_counts)

    @property
    def statistics(self) -> SmtStatistics:
        return self._stats


# ===================================================================
# PredicateRefinementEngine — counterexample-guided predicate refinement
# ===================================================================

@dataclass
class RefinementStep:
    """A single step in the contract discovery refinement loop."""
    iteration: int
    new_predicates: List[Predicate]
    counterexample: Optional[CfgPath] = None
    feasible: Optional[bool] = None
    interpolant: Optional[Predicate] = None


class PredicateRefinementEngine:
    """Counterexample-guided predicate refinement engine.

    Orchestrates the abstraction-checking-refinement loop (CEGAR-style).
    """

    def __init__(
        self,
        solver: Optional[SmtSolver] = None,
        *,
        max_iterations: int = 100,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._solver = solver or FallbackSolver()
        self._subtype_checker = SubtypeChecker(self._solver)
        self._ce_checker = CounterexampleChecker(self._solver)
        self._interpolant_extractor = InterpolantExtractor(self._solver)
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._history: List[RefinementStep] = []
        self._predicates: List[Predicate] = []

    @property
    def predicates(self) -> List[Predicate]:
        return list(self._predicates)

    @property
    def history(self) -> List[RefinementStep]:
        return list(self._history)

    @property
    def iteration_count(self) -> int:
        return len(self._history)

    def add_initial_predicates(self, preds: List[Predicate]) -> None:
        """Set the initial predicate set."""
        for p in preds:
            if p not in self._predicates:
                self._predicates.append(p)

    def refine_from_counterexample(
        self, counterexample: CfgPath
    ) -> RefinementStep:
        """Perform one refinement step from a counterexample.

        1. Check if the counterexample is feasible.
        2. If infeasible, extract an interpolant.
        3. Add the interpolant to the predicate set.
        """
        iteration = len(self._history)

        # Check feasibility
        feasibility = self._ce_checker.check_feasibility(counterexample)

        if feasibility.feasible:
            # Real bug — no new predicates
            step = RefinementStep(
                iteration=iteration,
                new_predicates=[],
                counterexample=counterexample,
                feasible=True,
            )
            self._history.append(step)
            return step

        # Infeasible — extract interpolant
        new_preds: List[Predicate] = []

        if feasibility.unsat_core is not None and len(feasibility.unsat_core.core) >= 2:
            # Use unsat core for predicate extraction
            core = feasibility.unsat_core.core
            mid = len(core) // 2
            a = And(tuple(core[:mid])) if mid > 1 else core[0]
            b = And(tuple(core[mid:])) if len(core) - mid > 1 else core[mid]

            interpolant = self._interpolant_extractor.extract_interpolant(a, b)
            if interpolant is not None:
                new_preds = self._extract_new_atoms(interpolant)

        # If no interpolant, try extracting predicates from path conditions
        if not new_preds:
            new_preds = self._extract_path_predicates(counterexample)

        # Add new predicates
        for p in new_preds:
            if p not in self._predicates:
                self._predicates.append(p)

        interpolant_pred = new_preds[0] if new_preds else None
        step = RefinementStep(
            iteration=iteration,
            new_predicates=new_preds,
            counterexample=counterexample,
            feasible=False,
            interpolant=interpolant_pred,
        )
        self._history.append(step)
        return step

    def _extract_new_atoms(self, pred: Predicate) -> List[Predicate]:
        """Extract atomic predicates from an interpolant."""
        atoms: List[Predicate] = []
        if isinstance(pred, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
            if pred not in self._predicates:
                atoms.append(pred)
        elif isinstance(pred, Not):
            if isinstance(pred.operand, (Comparison, IsInstance, IsNone, IsTruthy, HasAttr)):
                if pred.operand not in self._predicates:
                    atoms.append(pred.operand)
                if pred not in self._predicates:
                    atoms.append(pred)
        elif isinstance(pred, And):
            for c in pred.conjuncts:
                atoms.extend(self._extract_new_atoms(c))
        elif isinstance(pred, Or):
            for d in pred.disjuncts:
                atoms.extend(self._extract_new_atoms(d))
        return atoms

    def _extract_path_predicates(self, path: CfgPath) -> List[Predicate]:
        """Extract predicates from path conditions."""
        result: List[Predicate] = []
        for edge in path.edges:
            if edge.condition is not None:
                atoms = self._extract_new_atoms(edge.condition)
                result.extend(atoms)
        return result

    def check_convergence(self) -> bool:
        """Check if the refinement has converged (no new predicates in last step)."""
        if not self._history:
            return False
        last = self._history[-1]
        return len(last.new_predicates) == 0

    def summary(self) -> Dict[str, Any]:
        """Summary of the refinement process."""
        return {
            "iterations": len(self._history),
            "total_predicates": len(self._predicates),
            "converged": self.check_convergence(),
            "real_bugs_found": sum(
                1 for s in self._history if s.feasible is True
            ),
            "spurious_ce_eliminated": sum(
                1 for s in self._history if s.feasible is False
            ),
        }


# ===================================================================
# SmtPrettyPrinter — format SMT formulas for debugging
# ===================================================================

class SmtPrettyPrinter:
    """Pretty-print SMT formulas and results for debugging."""

    def format_predicate(self, pred: Predicate, *, indent: int = 0) -> str:
        prefix = "  " * indent
        if isinstance(pred, BoolLit):
            return f"{prefix}{'⊤' if pred.value else '⊥'}"
        if isinstance(pred, Comparison):
            return f"{prefix}{self._format_expr(pred.left)} {pred.op.value} {self._format_expr(pred.right)}"
        if isinstance(pred, IsInstance):
            return f"{prefix}isinstance({pred.var}, {pred.tag})"
        if isinstance(pred, IsNone):
            return f"{prefix}is_none({pred.var})"
        if isinstance(pred, IsTruthy):
            return f"{prefix}is_truthy({pred.var})"
        if isinstance(pred, HasAttr):
            return f"{prefix}hasattr({pred.var}, {pred.key!r})"
        if isinstance(pred, Not):
            return f"{prefix}¬({self.format_predicate(pred.operand)})"
        if isinstance(pred, And):
            if len(pred.conjuncts) <= 2:
                parts = " ∧ ".join(
                    self.format_predicate(c) for c in pred.conjuncts
                )
                return f"{prefix}{parts}"
            lines = [f"{prefix}∧"]
            for c in pred.conjuncts:
                lines.append(self.format_predicate(c, indent=indent + 1))
            return "\n".join(lines)
        if isinstance(pred, Or):
            if len(pred.disjuncts) <= 2:
                parts = " ∨ ".join(
                    self.format_predicate(d) for d in pred.disjuncts
                )
                return f"{prefix}{parts}"
            lines = [f"{prefix}∨"]
            for d in pred.disjuncts:
                lines.append(self.format_predicate(d, indent=indent + 1))
            return "\n".join(lines)
        if isinstance(pred, Implies):
            a = self.format_predicate(pred.antecedent)
            b = self.format_predicate(pred.consequent)
            return f"{prefix}{a} → {b}"
        if isinstance(pred, Iff):
            l = self.format_predicate(pred.left)
            r = self.format_predicate(pred.right)
            return f"{prefix}{l} ↔ {r}"
        return f"{prefix}{repr(pred)}"

    def _format_expr(self, expr: Expr) -> str:
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, Const):
            return repr(expr.value)
        if isinstance(expr, Len):
            return f"|{self._format_expr(expr.arg)}|"
        if isinstance(expr, BinOp):
            l = self._format_expr(expr.left)
            r = self._format_expr(expr.right)
            return f"({l} {expr.op.value} {r})"
        if isinstance(expr, UnaryOp):
            inner = self._format_expr(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return f"(-{inner})"
            if expr.op == UnaryArithOp.ABS:
                return f"|{inner}|"
        return repr(expr)

    def format_result(self, result: SmtQueryResult) -> str:
        """Format a query result for display."""
        lines = [f"Result: {result.result.value} ({result.time_seconds:.3f}s)"]
        if result.model is not None:
            lines.append("Model:")
            for name, value in sorted(result.model.variable_values.items()):
                lines.append(f"  {name} = {value!r}")
        if result.unsat_core is not None:
            lines.append(f"Unsat core: {len(result.unsat_core)} formulas")
            for label in result.unsat_core.labels:
                lines.append(f"  {label}")
        return "\n".join(lines)

    def format_typing_context(self, ctx: TypingContext) -> str:
        """Format a typing context for display."""
        lines = ["Typing Context:"]
        for name, ty in sorted(ctx.bindings.items()):
            lines.append(f"  {name}: {self._format_type(ty)}")
        if ctx.assumptions:
            lines.append("  Assumptions:")
            for a in ctx.assumptions:
                lines.append(f"    {self.format_predicate(a)}")
        return "\n".join(lines)

    def _format_type(self, ty: RefinementType) -> str:
        if isinstance(ty, BaseType):
            return ty.tag
        if isinstance(ty, RefinedType):
            return f"{{{ty.var}: {ty.base.tag} | {self.format_predicate(ty.refinement)}}}"
        if isinstance(ty, FunctionType):
            params = ", ".join(
                f"{n}: {self._format_type(t)}"
                for n, t in zip(ty.param_names, ty.param_types)
            )
            ret = self._format_type(ty.return_type)
            return f"({params}) → {ret}"
        if isinstance(ty, UnionType):
            return " | ".join(self._format_type(a) for a in ty.alternatives)
        if isinstance(ty, RecursiveType):
            return f"μ{ty.name}.{self._format_type(ty.body)}"
        return repr(ty)

    def format_refinement_step(self, step: RefinementStep) -> str:
        """Format a contract discovery refinement step."""
        lines = [f"Refinement Step {step.iteration}:"]
        if step.feasible is not None:
            lines.append(f"  Counterexample feasible: {step.feasible}")
        if step.interpolant is not None:
            lines.append(f"  Interpolant: {self.format_predicate(step.interpolant)}")
        if step.new_predicates:
            lines.append(f"  New predicates ({len(step.new_predicates)}):")
            for p in step.new_predicates:
                lines.append(f"    {self.format_predicate(p)}")
        else:
            lines.append("  No new predicates")
        return "\n".join(lines)
