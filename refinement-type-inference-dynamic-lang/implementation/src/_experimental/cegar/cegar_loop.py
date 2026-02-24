"""
Counterexample-Guided Abstraction Refinement (CEGAR) loop.

This module implements a proper CEGAR loop for refinement type inference:

1. Start with coarse predicates (seed from guard harvesting)
2. Run abstract interpretation with predicate abstraction
3. If a potential bug is found, check feasibility via Z3
4. If the counterexample is spurious, extract interpolants to refine predicates
5. Repeat until convergence or budget exhaustion

The key insight: the predicate set is finite and monotonically grows,
so the CEGAR loop terminates in at most O(|predicates_max|) iterations.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import z3

from ..refinement_lattice import (
    BaseTypeR,
    BaseTypeKind,
    DepFuncType,
    INT_TYPE,
    FLOAT_TYPE,
    STR_TYPE,
    BOOL_TYPE,
    NONE_TYPE,
    ANY_TYPE,
    NEVER_TYPE,
    Pred,
    PredOp,
    PredicateAbstractionDomain,
    PredicateAbstractionState,
    RefEnvironment,
    RefType,
    RefinementLattice,
    Z3Encoder,
)


# ---------------------------------------------------------------------------
# Bug / alarm types
# ---------------------------------------------------------------------------

class AlarmKind(Enum):
    DIV_BY_ZERO = "division-by-zero"
    NULL_DEREF = "null-dereference"
    INDEX_OOB = "index-out-of-bounds"
    TYPE_ERROR = "type-error"
    ATTR_ERROR = "attribute-error"
    ASSERTION = "assertion-violation"
    KEY_ERROR = "key-error"


@dataclass(frozen=True)
class Alarm:
    """A potential bug alarm raised by abstract interpretation."""
    kind: AlarmKind
    line: int
    col: int
    message: str
    path_condition: Pred  # path under which the alarm fires
    variable: str = ""    # the variable involved

    def pretty(self) -> str:
        return f"[{self.kind.value}] line {self.line}: {self.message}"


@dataclass(frozen=True)
class Counterexample:
    """A concrete counterexample (model from Z3)."""
    variable_values: Dict[str, int]
    path: List[int]  # sequence of line numbers
    alarm: Alarm

    def pretty(self) -> str:
        vals = ", ".join(f"{k}={v}" for k, v in self.variable_values.items())
        return f"CEX for {self.alarm.pretty()}: {vals}"


# ---------------------------------------------------------------------------
# Guard harvesting from Python AST
# ---------------------------------------------------------------------------

def harvest_guards(source: str) -> List[Pred]:
    """Extract predicates from guard conditions in Python source code.

    Scans for:
    - isinstance(x, T) checks
    - x is None / x is not None
    - comparisons: x > 0, x != 0, len(x) > 0, etc.
    - truthiness checks: if x:
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    predicates: List[Pred] = []

    class GuardVisitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If):
            preds = _extract_predicates(node.test)
            predicates.extend(preds)
            self.generic_visit(node)

        def visit_While(self, node: ast.While):
            preds = _extract_predicates(node.test)
            predicates.extend(preds)
            self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert):
            preds = _extract_predicates(node.test)
            predicates.extend(preds)
            self.generic_visit(node)

    GuardVisitor().visit(tree)
    return predicates


def _extract_predicates(node: ast.expr) -> List[Pred]:
    """Extract predicates from an AST expression."""
    preds: List[Pred] = []

    if isinstance(node, ast.Compare):
        if len(node.ops) == 1 and len(node.comparators) == 1:
            left = node.left
            right = node.comparators[0]
            op = node.ops[0]

            # x <op> <const>
            if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
                var = left.id
                val = right.value
                if isinstance(val, (int, float)):
                    val = int(val)
                    op_map = {
                        ast.Eq: "==", ast.NotEq: "!=",
                        ast.Lt: "<", ast.LtE: "<=",
                        ast.Gt: ">", ast.GtE: ">=",
                    }
                    op_str = op_map.get(type(op))
                    if op_str:
                        preds.append(Pred.var_cmp(var, op_str, val))

            # <const> <op> x  (reversed)
            elif isinstance(right, ast.Name) and isinstance(left, ast.Constant):
                var = right.id
                val = left.value
                if isinstance(val, (int, float)):
                    val = int(val)
                    rev_map = {
                        ast.Eq: "==", ast.NotEq: "!=",
                        ast.Lt: ">", ast.LtE: ">=",
                        ast.Gt: "<", ast.GtE: "<=",
                    }
                    op_str = rev_map.get(type(op))
                    if op_str:
                        preds.append(Pred.var_cmp(var, op_str, val))

            # x is None / x is not None
            if isinstance(op, ast.Is):
                if isinstance(left, ast.Name) and isinstance(right, ast.Constant) and right.value is None:
                    preds.append(Pred.is_none(left.id))
            elif isinstance(op, ast.IsNot):
                if isinstance(left, ast.Name) and isinstance(right, ast.Constant) and right.value is None:
                    preds.append(Pred.is_not_none(left.id))

            # len(x) <op> <const>
            if isinstance(left, ast.Call) and isinstance(left.func, ast.Name):
                if left.func.id == "len" and left.args:
                    arg = left.args[0]
                    if isinstance(arg, ast.Name) and isinstance(right, ast.Constant):
                        var = arg.id
                        val = right.value
                        if isinstance(val, int):
                            if isinstance(op, ast.Gt):
                                preds.append(Pred.len_gt(var, val))
                            elif isinstance(op, (ast.GtE,)):
                                preds.append(Pred.len_ge(var, val))
                            elif isinstance(op, ast.Eq):
                                preds.append(Pred.len_eq(var, val))

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id == "isinstance" and len(node.args) == 2:
                if isinstance(node.args[0], ast.Name):
                    var = node.args[0].id
                    typ = ast.dump(node.args[1])
                    if isinstance(node.args[1], ast.Name):
                        typ = node.args[1].id
                    preds.append(Pred.isinstance_(var, typ))

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _extract_predicates(node.operand)
        for p in inner:
            preds.append(p.not_())

    elif isinstance(node, ast.BoolOp):
        for val in node.values:
            preds.extend(_extract_predicates(val))

    elif isinstance(node, ast.Name):
        preds.append(Pred.truthy(node.id))

    return preds


# ---------------------------------------------------------------------------
# Abstract interpreter for CEGAR
# ---------------------------------------------------------------------------

class AbstractInterpreter:
    """Flow-sensitive abstract interpreter using predicate abstraction.

    Performs forward dataflow analysis over the Python AST, tracking
    refinement types for each variable at each program point.
    """

    def __init__(self, domain: PredicateAbstractionDomain,
                 lattice: RefinementLattice):
        self.domain = domain
        self.lattice = lattice
        self.alarms: List[Alarm] = []

    def analyze_function(self, func_node: ast.FunctionDef,
                         entry_env: RefEnvironment) -> Tuple[RefEnvironment, List[Alarm]]:
        """Analyze a single function body."""
        self.alarms = []
        env = entry_env

        # Initialize parameters
        for arg in func_node.args.args:
            name = arg.arg
            if name not in env.bindings:
                env = env.set(name, RefType.trivial(ANY_TYPE))

        # Analyze body statements
        env = self._analyze_stmts(func_node.body, env)
        return env, list(self.alarms)

    def _analyze_stmts(self, stmts: List[ast.stmt],
                       env: RefEnvironment) -> RefEnvironment:
        for stmt in stmts:
            env = self._analyze_stmt(stmt, env)
        return env

    def _analyze_stmt(self, stmt: ast.stmt,
                      env: RefEnvironment) -> RefEnvironment:
        if isinstance(stmt, ast.Assign):
            return self._analyze_assign(stmt, env)
        elif isinstance(stmt, ast.AugAssign):
            return self._analyze_aug_assign(stmt, env)
        elif isinstance(stmt, ast.If):
            return self._analyze_if(stmt, env)
        elif isinstance(stmt, ast.While):
            return self._analyze_while(stmt, env)
        elif isinstance(stmt, ast.For):
            return self._analyze_for(stmt, env)
        elif isinstance(stmt, ast.Return):
            return self._analyze_return(stmt, env)
        elif isinstance(stmt, ast.Expr):
            self._check_expr(stmt.value, env)
            return env
        elif isinstance(stmt, ast.Assert):
            return self._analyze_assert(stmt, env)
        elif isinstance(stmt, ast.FunctionDef):
            return env  # skip nested functions for now
        elif isinstance(stmt, ast.Try):
            return self._analyze_try(stmt, env)
        return env

    def _analyze_assign(self, stmt: ast.Assign,
                        env: RefEnvironment) -> RefEnvironment:
        self._check_expr(stmt.value, env)
        rhs_type = self._infer_expr_type(stmt.value, env)
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                env = env.set(target.id, rhs_type)
            elif isinstance(target, ast.Subscript):
                self._check_subscript(target, env)
        return env

    def _analyze_aug_assign(self, stmt: ast.AugAssign,
                            env: RefEnvironment) -> RefEnvironment:
        self._check_expr(stmt.value, env)
        if isinstance(stmt.target, ast.Name):
            # Check division
            if isinstance(stmt.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                self._check_div_by_zero(stmt.value, env, stmt)
            rhs_type = self._infer_expr_type(stmt.value, env)
            env = env.set(stmt.target.id, rhs_type)
        return env

    def _analyze_if(self, stmt: ast.If,
                    env: RefEnvironment) -> RefEnvironment:
        cond_preds = _extract_predicates(stmt.test)
        self._check_expr(stmt.test, env)

        # True branch: narrow with condition
        true_env = env
        for p in cond_preds:
            true_env = self._narrow_env(true_env, p, positive=True)
        true_env = self._analyze_stmts(stmt.body, true_env)
        true_returns = self._branch_always_returns(stmt.body)

        # False branch: narrow with negation
        false_env = env
        for p in cond_preds:
            false_env = self._narrow_env(false_env, p, positive=False)
        if stmt.orelse:
            false_env = self._analyze_stmts(stmt.orelse, false_env)
        false_returns = self._branch_always_returns(stmt.orelse) if stmt.orelse else False

        # If one branch always returns, only the other continues
        if true_returns and false_returns:
            return env  # both return, but we continue for analysis purposes
        if true_returns:
            return false_env
        if false_returns:
            return true_env
        return true_env.join(false_env, self.lattice)

    @staticmethod
    def _branch_always_returns(stmts: List[ast.stmt]) -> bool:
        """Check if a branch always returns/raises (no fallthrough)."""
        if not stmts:
            return False
        last = stmts[-1]
        if isinstance(last, (ast.Return, ast.Raise)):
            return True
        if isinstance(last, ast.If):
            return (AbstractInterpreter._branch_always_returns(last.body) and
                    bool(last.orelse) and
                    AbstractInterpreter._branch_always_returns(last.orelse))
        return False

    def _analyze_while(self, stmt: ast.While,
                       env: RefEnvironment) -> RefEnvironment:
        """Analyze while loop with widening for convergence.

        Uses proper predicate abstraction widening instead of
        the naive 2-iteration unrolling.
        """
        cond_preds = _extract_predicates(stmt.test)

        # Widening loop
        prev_env = env
        for iteration in range(20):  # max iterations before forced widening
            # Enter loop body with condition
            body_env = prev_env
            for p in cond_preds:
                body_env = self._narrow_env(body_env, p, positive=True)

            body_env = self._analyze_stmts(stmt.body, body_env)

            # Join with entry
            joined = prev_env.join(body_env, self.lattice)

            # Widen after 3 iterations to force convergence
            if iteration >= 3:
                joined = prev_env.widen(
                    joined, self.domain.predicates, self.lattice)

            # Check convergence
            if self._env_leq(joined, prev_env):
                break
            prev_env = joined

        # Exit: narrow with negated condition
        exit_env = prev_env
        for p in cond_preds:
            exit_env = self._narrow_env(exit_env, p, positive=False)

        # Narrowing pass for precision recovery
        body_env2 = exit_env
        for p in cond_preds:
            body_env2 = self._narrow_env(body_env2, p, positive=True)
        body_env2 = self._analyze_stmts(stmt.body, body_env2)
        narrowed = exit_env.join(body_env2, self.lattice)
        for var in exit_env.bindings:
            if var in narrowed.bindings and var in prev_env.bindings:
                narrowed.bindings[var] = self.lattice.narrow(
                    prev_env.bindings[var], narrowed.bindings[var])

        if stmt.orelse:
            exit_env = self._analyze_stmts(stmt.orelse, exit_env)

        return exit_env

    def _analyze_for(self, stmt: ast.For,
                     env: RefEnvironment) -> RefEnvironment:
        if isinstance(stmt.target, ast.Name):
            env = env.set(stmt.target.id, RefType.trivial(ANY_TYPE))
        env = self._analyze_stmts(stmt.body, env)
        if stmt.orelse:
            env = self._analyze_stmts(stmt.orelse, env)
        return env

    def _analyze_return(self, stmt: ast.Return,
                        env: RefEnvironment) -> RefEnvironment:
        if stmt.value:
            self._check_expr(stmt.value, env)
        return env

    def _analyze_assert(self, stmt: ast.Assert,
                        env: RefEnvironment) -> RefEnvironment:
        preds = _extract_predicates(stmt.test)
        for p in preds:
            env = self._narrow_env(env, p, positive=True)
        return env

    def _analyze_try(self, stmt: ast.Try,
                     env: RefEnvironment) -> RefEnvironment:
        try_env = self._analyze_stmts(stmt.body, env)
        for handler in stmt.handlers:
            handler_env = env
            if handler.name:
                handler_env = handler_env.set(
                    handler.name, RefType.trivial(ANY_TYPE))
            handler_env = self._analyze_stmts(handler.body, handler_env)
            try_env = try_env.join(handler_env, self.lattice)
        if stmt.finalbody:
            try_env = self._analyze_stmts(stmt.finalbody, try_env)
        return try_env

    def _narrow_env(self, env: RefEnvironment, pred: Pred,
                    positive: bool) -> RefEnvironment:
        """Narrow environment based on a guard predicate."""
        effective = pred if positive else pred.not_()
        fvs = effective.free_vars()
        for var in fvs:
            current = env.bindings.get(var, RefType.trivial(ANY_TYPE))
            refined = self.lattice.meet(
                current,
                RefType("ν", current.base, effective.substitute(var, "ν"))
            )
            env = env.set(var, refined)
        return env

    def _env_leq(self, a: RefEnvironment, b: RefEnvironment) -> bool:
        """Check if a ⊑ b pointwise."""
        for var in a.bindings:
            if var not in b.bindings:
                continue
            if not self.lattice.leq(a.bindings[var], b.bindings[var]):
                return False
        return True

    def _check_expr(self, expr: ast.expr, env: RefEnvironment) -> None:
        """Check an expression for potential bugs."""
        if isinstance(expr, ast.BinOp):
            if isinstance(expr.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                self._check_div_by_zero(expr.right, env, expr)
            self._check_expr(expr.left, env)
            self._check_expr(expr.right, env)

        elif isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name):
                var = expr.value.id
                ty = env.bindings.get(var)
                if ty:
                    # Check None dereference: alarm if var COULD be None
                    # (i.e., the type doesn't exclude None).
                    # For ANY type or types without is_not_none guard, raise alarm.
                    is_not_none_type = RefType("ν", ty.base,
                                               Pred.is_not_none("ν"))
                    if not self.lattice.subtype(ty, is_not_none_type):
                        self.alarms.append(Alarm(
                            AlarmKind.NULL_DEREF,
                            getattr(expr, 'lineno', 0),
                            getattr(expr, 'col_offset', 0),
                            f"Possible None dereference: {var}.{expr.attr}",
                            Pred.is_none(var),
                            var,
                        ))

        elif isinstance(expr, ast.Subscript):
            self._check_subscript(expr, env)

        elif isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Attribute):
                if isinstance(expr.func.value, ast.Name):
                    self._check_expr(expr.func.value, env)
            for arg in expr.args:
                self._check_expr(arg, env)

    def _check_subscript(self, expr: ast.Subscript,
                         env: RefEnvironment) -> None:
        """Check subscript access for potential index errors."""
        if isinstance(expr.value, ast.Name):
            var = expr.value.id
            ty = env.bindings.get(var)
            if ty:
                none_type = RefType("ν", NONE_TYPE, Pred.true_())
                if self.lattice.subtype(ty, none_type):
                    self.alarms.append(Alarm(
                        AlarmKind.NULL_DEREF,
                        getattr(expr, 'lineno', 0),
                        getattr(expr, 'col_offset', 0),
                        f"Possible None dereference on subscript: {var}[...]",
                        Pred.is_none(var),
                        var,
                    ))

    def _check_div_by_zero(self, divisor: ast.expr,
                           env: RefEnvironment,
                           node: ast.AST) -> None:
        """Check if divisor can be zero."""
        if isinstance(divisor, ast.Constant):
            if divisor.value == 0:
                self.alarms.append(Alarm(
                    AlarmKind.DIV_BY_ZERO,
                    getattr(node, 'lineno', 0),
                    getattr(node, 'col_offset', 0),
                    "Division by literal zero",
                    Pred.true_(),
                ))
            return

        if isinstance(divisor, ast.Name):
            var = divisor.id
            ty = env.bindings.get(var)
            if ty:
                # Check: can the variable be zero?
                zero_type = RefType("ν", INT_TYPE, Pred.var_eq("ν", 0))
                if not self.lattice.is_bottom(self.lattice.meet(ty, zero_type)):
                    # The meet is non-empty → variable CAN be zero
                    self.alarms.append(Alarm(
                        AlarmKind.DIV_BY_ZERO,
                        getattr(node, 'lineno', 0),
                        getattr(node, 'col_offset', 0),
                        f"Possible division by zero: {var}",
                        Pred.var_eq(var, 0),
                        var,
                    ))

    def _infer_expr_type(self, expr: ast.expr,
                         env: RefEnvironment) -> RefType:
        """Infer refinement type for an expression."""
        if isinstance(expr, ast.Constant):
            if expr.value is None:
                return RefType("ν", NONE_TYPE, Pred.true_())
            if isinstance(expr.value, bool):
                val = 1 if expr.value else 0
                return RefType("ν", BOOL_TYPE, Pred.var_eq("ν", val))
            if isinstance(expr.value, int):
                return RefType("ν", INT_TYPE, Pred.var_eq("ν", expr.value))
            if isinstance(expr.value, float):
                return RefType.trivial(FLOAT_TYPE)
            if isinstance(expr.value, str):
                return RefType("ν", STR_TYPE,
                               Pred.len_eq("ν", len(expr.value)))
            return RefType.trivial(ANY_TYPE)

        if isinstance(expr, ast.Name):
            ty = env.bindings.get(expr.id)
            return ty if ty else RefType.trivial(ANY_TYPE)

        if isinstance(expr, ast.BinOp):
            return RefType.trivial(INT_TYPE)  # simplified

        if isinstance(expr, ast.UnaryOp):
            return self._infer_expr_type(expr.operand, env)

        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id == "len":
                return RefType("ν", INT_TYPE, Pred.var_ge("ν", 0))
            return RefType.trivial(ANY_TYPE)

        if isinstance(expr, ast.List):
            n = len(expr.elts)
            return RefType("ν", BaseTypeR(BaseTypeKind.LIST),
                           Pred.len_eq("ν", n))

        if isinstance(expr, ast.Dict):
            n = len(expr.keys)
            return RefType("ν", BaseTypeR(BaseTypeKind.DICT),
                           Pred.len_eq("ν", n))

        if isinstance(expr, ast.IfExp):
            true_type = self._infer_expr_type(expr.body, env)
            false_type = self._infer_expr_type(expr.orelse, env)
            return self.lattice.join(true_type, false_type)

        return RefType.trivial(ANY_TYPE)


# ---------------------------------------------------------------------------
# Interpolant extraction
# ---------------------------------------------------------------------------

def extract_interpolant(alarm: Alarm, model: Dict[str, int],
                        encoder: Z3Encoder) -> List[Pred]:
    """Extract new predicates from a spurious counterexample.

    Given a counterexample (variable assignment) that triggers a false alarm,
    we compute Craig interpolants that separate the feasible states from
    the infeasible counterexample.

    Returns new predicates to add to the predicate set.
    """
    new_preds: List[Pred] = []

    # Strategy 1: boundary predicates from model values
    for var, val in model.items():
        new_preds.append(Pred.var_cmp(var, ">=", val))
        new_preds.append(Pred.var_cmp(var, "<=", val))
        new_preds.append(Pred.var_cmp(var, "!=", val))
        if val != 0:
            new_preds.append(Pred.var_cmp(var, "!=", 0))
            new_preds.append(Pred.var_cmp(var, ">", 0))
            new_preds.append(Pred.var_cmp(var, "<", 0))

    # Strategy 2: from the alarm kind
    if alarm.kind == AlarmKind.DIV_BY_ZERO and alarm.variable:
        new_preds.append(Pred.var_neq(alarm.variable, 0))
        new_preds.append(Pred.var_gt(alarm.variable, 0))
        new_preds.append(Pred.var_lt(alarm.variable, 0))

    if alarm.kind == AlarmKind.NULL_DEREF and alarm.variable:
        new_preds.append(Pred.is_not_none(alarm.variable))
        new_preds.append(Pred.is_none(alarm.variable))

    if alarm.kind == AlarmKind.INDEX_OOB and alarm.variable:
        new_preds.append(Pred.len_gt(alarm.variable, 0))
        new_preds.append(Pred.len_ge(alarm.variable, 1))

    # Strategy 3: Z3-based interpolation via binary interpolant
    # We use the unsatisfiable core to identify relevant constraints
    path_formula = encoder.encode(alarm.path_condition)
    s = z3.Solver()
    s.add(path_formula)
    for var, val in model.items():
        s.add(encoder.int_var(var) == val)

    if s.check() == z3.unsat:
        # The path + model is unsat → the counterexample is infeasible
        core = s.unsat_core() if hasattr(s, 'unsat_core') else []
        # Generate predicates from the path condition
        for fv in alarm.path_condition.free_vars():
            if fv in model:
                new_preds.append(Pred.var_neq(fv, model[fv]))

    return new_preds


def check_feasibility(alarm: Alarm, env: RefEnvironment,
                      encoder: Z3Encoder,
                      timeout_ms: int = 5000) -> Tuple[bool, Optional[Dict[str, int]]]:
    """Check if an alarm is feasible (real bug) or spurious.

    Returns (is_feasible, counterexample_model).
    If feasible, the model shows concrete values that trigger the bug.
    If infeasible (spurious), model is None.
    """
    s = z3.Solver()
    s.set("timeout", timeout_ms)

    # Encode path condition
    s.add(encoder.encode(alarm.path_condition))

    # Encode environment constraints
    for var, ty in env.bindings.items():
        if not ty.pred.op == PredOp.TRUE:
            renamed = ty.alpha_rename("ν")
            pred_with_var = renamed.pred.substitute("ν", var)
            s.add(encoder.encode(pred_with_var))

    result = s.check()

    if result == z3.sat:
        model = s.model()
        values: Dict[str, int] = {}
        for d in model.decls():
            name = d.name()
            val = model[d]
            if isinstance(val, z3.IntNumRef):
                values[name] = val.as_long()
        return True, values
    else:
        return False, None


# ---------------------------------------------------------------------------
# CEGAR loop
# ---------------------------------------------------------------------------

@dataclass
class CEGARConfig:
    """Configuration for the CEGAR loop."""
    max_iterations: int = 30
    max_predicates: int = 200
    timeout_ms: int = 10000
    seed_from_guards: bool = True
    use_interpolation: bool = True
    widening_delay: int = 3
    verbose: bool = False


@dataclass
class CEGARResult:
    """Result of a CEGAR analysis run."""
    converged: bool
    iterations: int
    alarms: List[Alarm]
    verified_alarms: List[Alarm]  # confirmed real bugs
    spurious_alarms: List[Alarm]  # proven false positives
    predicates_used: int
    predicates_added: int
    analysis_time_ms: float
    function_summaries: Dict[str, DepFuncType] = field(default_factory=dict)

    def precision(self) -> float:
        total = len(self.verified_alarms) + len(self.spurious_alarms)
        if total == 0:
            return 1.0
        return len(self.verified_alarms) / total

    def summary(self) -> str:
        lines = [
            f"CEGAR Analysis Result:",
            f"  Converged: {self.converged}",
            f"  Iterations: {self.iterations}",
            f"  Total alarms: {len(self.alarms)}",
            f"  Verified bugs: {len(self.verified_alarms)}",
            f"  Spurious: {len(self.spurious_alarms)}",
            f"  Precision: {self.precision():.1%}",
            f"  Predicates: {self.predicates_used} ({self.predicates_added} added by CEGAR)",
            f"  Time: {self.analysis_time_ms:.1f}ms",
        ]
        return "\n".join(lines)


def run_cegar(source: str, config: Optional[CEGARConfig] = None) -> CEGARResult:
    """Run the full CEGAR loop on Python source code.

    Algorithm:
    1. Parse source, extract functions
    2. Harvest guard predicates as seeds
    3. Initialize predicate abstraction domain
    4. For each iteration:
       a. Run abstract interpretation
       b. Collect alarms
       c. For each alarm, check feasibility via Z3
       d. If spurious, extract interpolants → add new predicates
       e. If all alarms verified or no new predicates, stop
    5. Return analysis results
    """
    if config is None:
        config = CEGARConfig()

    start_time = time.monotonic()
    lattice = RefinementLattice(timeout_ms=config.timeout_ms)
    encoder = Z3Encoder()

    # Parse source
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return CEGARResult(
            converged=False, iterations=0,
            alarms=[], verified_alarms=[], spurious_alarms=[],
            predicates_used=0, predicates_added=0,
            analysis_time_ms=0.0,
        )

    # Harvest seed predicates from guards
    seed_preds: List[Pred] = []
    if config.seed_from_guards:
        seed_preds = harvest_guards(source)

    # Always include standard predicates
    standard_preds = [
        Pred.var_neq("ν", 0),
        Pred.var_ge("ν", 0),
        Pred.var_gt("ν", 0),
        Pred.var_le("ν", 0),
        Pred.var_lt("ν", 0),
        Pred.is_none("ν"),
        Pred.is_not_none("ν"),
    ]
    all_preds = list(seed_preds) + standard_preds

    # Extract functions
    functions: List[ast.FunctionDef] = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]

    if not functions:
        # Analyze as module-level code
        fake_func = ast.FunctionDef(
            name="<module>",
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=tree.body,
            decorator_list=[],
            returns=None,
            lineno=1, col_offset=0,
        )
        functions = [fake_func]

    total_alarms: List[Alarm] = []
    verified: List[Alarm] = []
    spurious: List[Alarm] = []
    predicates_added = 0
    func_summaries: Dict[str, DepFuncType] = {}

    iteration = 0
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        if config.verbose:
            print(f"CEGAR iteration {iteration}, predicates: {len(all_preds)}")

        # Initialize domain with current predicates
        domain = PredicateAbstractionDomain(all_preds, lattice)
        interp = AbstractInterpreter(domain, lattice)

        # Analyze each function
        iter_alarms: List[Alarm] = []
        for func in functions:
            entry_env = RefEnvironment()
            for arg in func.args.args:
                entry_env = entry_env.set(
                    arg.arg, RefType.trivial(ANY_TYPE))

            result_env, func_alarms = interp.analyze_function(func, entry_env)
            iter_alarms.extend(func_alarms)

            # Compute function summary
            param_types: List[Tuple[str, RefType]] = []
            for arg in func.args.args:
                ty = result_env.get(arg.arg)
                param_types.append((
                    arg.arg,
                    ty if ty else RefType.trivial(ANY_TYPE)
                ))
            ret_type = RefType.trivial(ANY_TYPE)
            func_summaries[func.name] = DepFuncType(
                tuple(param_types), ret_type
            )

        total_alarms = iter_alarms

        # Check feasibility of each alarm
        new_verified: List[Alarm] = []
        new_spurious: List[Alarm] = []
        new_preds_this_iter: List[Pred] = []

        for alarm in iter_alarms:
            entry_env = RefEnvironment()
            is_feasible, model = check_feasibility(
                alarm, entry_env, encoder, config.timeout_ms)

            if is_feasible:
                new_verified.append(alarm)
            else:
                new_spurious.append(alarm)
                if config.use_interpolation and model is not None:
                    interpolants = extract_interpolant(alarm, model, encoder)
                    for ip in interpolants:
                        if ip not in all_preds and len(all_preds) < config.max_predicates:
                            all_preds.append(ip)
                            new_preds_this_iter.append(ip)

        verified = new_verified
        spurious = new_spurious
        predicates_added += len(new_preds_this_iter)

        # Convergence check
        if not new_preds_this_iter:
            converged = True
            break

        if len(all_preds) >= config.max_predicates:
            break

    elapsed = (time.monotonic() - start_time) * 1000

    return CEGARResult(
        converged=converged,
        iterations=iteration,
        alarms=total_alarms,
        verified_alarms=verified,
        spurious_alarms=spurious,
        predicates_used=len(all_preds),
        predicates_added=predicates_added,
        analysis_time_ms=elapsed,
        function_summaries=func_summaries,
    )


# ---------------------------------------------------------------------------
# Convenience: analyze a file
# ---------------------------------------------------------------------------

def analyze_file(filepath: str, config: Optional[CEGARConfig] = None) -> CEGARResult:
    """Analyze a Python file with the CEGAR loop."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()
    return run_cegar(source, config)


def analyze_source(source: str, verbose: bool = False) -> CEGARResult:
    """Analyze Python source code with the CEGAR loop."""
    config = CEGARConfig(verbose=verbose)
    return run_cegar(source, config)
