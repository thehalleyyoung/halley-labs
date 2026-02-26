"""
TLA-lite expression evaluator.

``evaluate(expr, env, state)`` reduces an AST expression node to a
``TLAValue`` given the current environment bindings and a concrete state
(for reading variable values).

The evaluator is a large pattern-match over AST node kinds.  Because the
parser/AST layer is still being built, this module defines its own
lightweight ``Expr`` node hierarchy.  Once the parser stabilises, these
nodes will be replaced by the canonical AST types.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
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

from .values import (
    TLAValue,
    TLAValueError,
    IntValue,
    BoolValue,
    StringValue,
    SetValue,
    FunctionValue,
    TupleValue,
    RecordValue,
    SequenceValue,
    ModelValue,
    value_from_python,
)
from .environment import Environment, OpDef
from .state import TLAState


# ===================================================================
# AST node definitions (lightweight, frozen dataclasses)
# ===================================================================

class ExprKind(Enum):
    INT_LIT = auto()
    BOOL_LIT = auto()
    STRING_LIT = auto()
    NAME_REF = auto()
    PRIMED_REF = auto()
    UNARY_OP = auto()
    BINARY_OP = auto()
    IF_THEN_ELSE = auto()
    LET_IN = auto()
    CASE = auto()
    QUANT_FORALL = auto()
    QUANT_EXISTS = auto()
    CHOOSE = auto()
    SET_ENUM = auto()
    SET_COMP = auto()
    SET_FILTER = auto()
    FUNC_CONSTRUCT = auto()
    FUNC_APPLY = auto()
    FUNC_EXCEPT = auto()
    DOMAIN_OP = auto()
    TUPLE_CONSTRUCT = auto()
    RECORD_CONSTRUCT = auto()
    RECORD_ACCESS = auto()
    RECORD_EXCEPT = auto()
    SEQ_OP = auto()
    OP_APPLY = auto()
    BUILTIN_CALL = auto()
    UNCHANGED = auto()


@dataclass(frozen=True)
class Expr:
    """Minimal AST expression node."""
    kind: ExprKind
    data: Any = None
    children: Tuple["Expr", ...] = ()

    def __repr__(self) -> str:
        return f"Expr({self.kind.name}, {self.data!r}, children={len(self.children)})"


# ===================================================================
# Convenience AST constructors
# ===================================================================

def int_lit(n: int) -> Expr:
    return Expr(ExprKind.INT_LIT, n)

def bool_lit(b: bool) -> Expr:
    return Expr(ExprKind.BOOL_LIT, b)

def string_lit(s: str) -> Expr:
    return Expr(ExprKind.STRING_LIT, s)

def name_ref(name: str) -> Expr:
    return Expr(ExprKind.NAME_REF, name)

def primed_ref(name: str) -> Expr:
    return Expr(ExprKind.PRIMED_REF, name)

def unary_op(op: str, operand: Expr) -> Expr:
    return Expr(ExprKind.UNARY_OP, op, (operand,))

def binary_op(op: str, left: Expr, right: Expr) -> Expr:
    return Expr(ExprKind.BINARY_OP, op, (left, right))

def if_then_else(cond: Expr, then_: Expr, else_: Expr) -> Expr:
    return Expr(ExprKind.IF_THEN_ELSE, None, (cond, then_, else_))

def let_in(defs: Tuple[Tuple[str, Expr], ...], body: Expr) -> Expr:
    return Expr(ExprKind.LET_IN, defs, (body,))

def case_expr(arms: Tuple[Tuple[Expr, Expr], ...], other: Optional[Expr] = None) -> Expr:
    return Expr(ExprKind.CASE, {"arms": arms, "other": other})

def quant_forall(var: str, domain: Expr, body: Expr) -> Expr:
    return Expr(ExprKind.QUANT_FORALL, var, (domain, body))

def quant_exists(var: str, domain: Expr, body: Expr) -> Expr:
    return Expr(ExprKind.QUANT_EXISTS, var, (domain, body))

def choose(var: str, domain: Expr, pred: Expr) -> Expr:
    return Expr(ExprKind.CHOOSE, var, (domain, pred))

def set_enum(*elements: Expr) -> Expr:
    return Expr(ExprKind.SET_ENUM, None, tuple(elements))

def set_comp(var: str, domain: Expr, map_expr: Expr) -> Expr:
    return Expr(ExprKind.SET_COMP, var, (domain, map_expr))

def set_filter(var: str, domain: Expr, pred: Expr) -> Expr:
    return Expr(ExprKind.SET_FILTER, var, (domain, pred))

def func_construct(var: str, domain: Expr, body: Expr) -> Expr:
    return Expr(ExprKind.FUNC_CONSTRUCT, var, (domain, body))

def func_apply(func: Expr, arg: Expr) -> Expr:
    return Expr(ExprKind.FUNC_APPLY, None, (func, arg))

def func_except(func: Expr, key: Expr, val: Expr) -> Expr:
    return Expr(ExprKind.FUNC_EXCEPT, None, (func, key, val))

def domain_op(expr: Expr) -> Expr:
    return Expr(ExprKind.DOMAIN_OP, None, (expr,))

def tuple_construct(*elements: Expr) -> Expr:
    return Expr(ExprKind.TUPLE_CONSTRUCT, None, tuple(elements))

def record_construct(fields: Tuple[Tuple[str, Expr], ...]) -> Expr:
    return Expr(ExprKind.RECORD_CONSTRUCT, fields)

def record_access(record: Expr, field_name: str) -> Expr:
    return Expr(ExprKind.RECORD_ACCESS, field_name, (record,))

def record_except(record: Expr, field_name: str, val: Expr) -> Expr:
    return Expr(ExprKind.RECORD_EXCEPT, field_name, (record, val))

def seq_op(op_name: str, *args: Expr) -> Expr:
    return Expr(ExprKind.SEQ_OP, op_name, tuple(args))

def op_apply(name: str, *args: Expr) -> Expr:
    return Expr(ExprKind.OP_APPLY, name, tuple(args))

def builtin_call(module: str, name: str, *args: Expr) -> Expr:
    return Expr(ExprKind.BUILTIN_CALL, (module, name), tuple(args))

def unchanged(*vars: str) -> Expr:
    return Expr(ExprKind.UNCHANGED, tuple(vars))


# ===================================================================
# Evaluation error with trace
# ===================================================================

class EvalError(TLAValueError):
    """Raised on evaluation failures, carrying a trace of contexts."""

    def __init__(self, msg: str, trace: List[str] | None = None) -> None:
        self.trace = trace or []
        full = msg
        if self.trace:
            full += "\n  Evaluation trace:\n    " + "\n    ".join(self.trace)
        super().__init__(full)

    def with_context(self, ctx: str) -> "EvalError":
        return EvalError(str(self), [ctx] + self.trace)


# ===================================================================
# Core evaluator
# ===================================================================

def evaluate(expr: Expr, env: Environment, state: TLAState) -> TLAValue:
    """Evaluate *expr* in the given environment and state.

    Dispatches on ``expr.kind`` to the appropriate handler.
    """
    try:
        return _eval(expr, env, state)
    except EvalError:
        raise
    except TLAValueError as exc:
        raise EvalError(str(exc), [f"in {expr.kind.name}"]) from exc


def _eval(expr: Expr, env: Environment, state: TLAState) -> TLAValue:
    kind = expr.kind

    # --- literals ---------------------------------------------------------
    if kind is ExprKind.INT_LIT:
        return IntValue(expr.data)

    if kind is ExprKind.BOOL_LIT:
        return BoolValue(expr.data)

    if kind is ExprKind.STRING_LIT:
        return StringValue(expr.data)

    # --- name reference ---------------------------------------------------
    if kind is ExprKind.NAME_REF:
        name: str = expr.data
        # first: local / constant bindings
        val = env.lookup(name)
        if val is not None:
            return val
        cval = env.constant_value(name)
        if cval is not None:
            return cval
        # second: state variable
        if state.has_var(name):
            return state.get(name)
        raise EvalError(f"Unresolved name '{name}'", [f"NAME_REF({name})"])

    # --- primed variable reference (next state) --------------------------
    if kind is ExprKind.PRIMED_REF:
        name = expr.data
        primed = name + "'"
        val = env.lookup(primed)
        if val is not None:
            return val
        raise EvalError(
            f"Primed variable '{name}' not bound in action context",
            [f"PRIMED_REF({name})"],
        )

    # --- unary operators --------------------------------------------------
    if kind is ExprKind.UNARY_OP:
        return _eval_unary(expr.data, expr.children[0], env, state)

    # --- binary operators -------------------------------------------------
    if kind is ExprKind.BINARY_OP:
        return _eval_binary(expr.data, expr.children[0], expr.children[1], env, state)

    # --- IF / THEN / ELSE (lazy) -----------------------------------------
    if kind is ExprKind.IF_THEN_ELSE:
        cond_val = _eval_bool(expr.children[0], env, state, "IF condition")
        if cond_val:
            return _eval(expr.children[1], env, state)
        else:
            return _eval(expr.children[2], env, state)

    # --- LET / IN --------------------------------------------------------
    if kind is ExprKind.LET_IN:
        defs: Tuple[Tuple[str, Expr], ...] = expr.data
        with env.scope("let"):
            for def_name, def_body in defs:
                env.bind(def_name, _eval(def_body, env, state))
            return _eval(expr.children[0], env, state)

    # --- CASE ------------------------------------------------------------
    if kind is ExprKind.CASE:
        return _eval_case(expr.data, env, state)

    # --- quantifiers -----------------------------------------------------
    if kind is ExprKind.QUANT_FORALL:
        return _eval_forall(expr.data, expr.children[0], expr.children[1], env, state)

    if kind is ExprKind.QUANT_EXISTS:
        return _eval_exists(expr.data, expr.children[0], expr.children[1], env, state)

    # --- CHOOSE ----------------------------------------------------------
    if kind is ExprKind.CHOOSE:
        return _eval_choose(expr.data, expr.children[0], expr.children[1], env, state)

    # --- set enumeration {a, b, c} ----------------------------------------
    if kind is ExprKind.SET_ENUM:
        elems = [_eval(c, env, state) for c in expr.children]
        return SetValue(elems)

    # --- set comprehension {e(x) : x \in S} ------------------------------
    if kind is ExprKind.SET_COMP:
        return _eval_set_comp(expr.data, expr.children[0], expr.children[1], env, state)

    # --- set filter {x \in S : P(x)} -------------------------------------
    if kind is ExprKind.SET_FILTER:
        return _eval_set_filter(expr.data, expr.children[0], expr.children[1], env, state)

    # --- function construction [x \in S |-> e(x)] -----------------------
    if kind is ExprKind.FUNC_CONSTRUCT:
        return _eval_func_construct(expr.data, expr.children[0], expr.children[1], env, state)

    # --- function application f[a] ----------------------------------------
    if kind is ExprKind.FUNC_APPLY:
        func_val = _eval(expr.children[0], env, state)
        arg_val = _eval(expr.children[1], env, state)
        return _apply_function(func_val, arg_val)

    # --- function EXCEPT --------------------------------------------------
    if kind is ExprKind.FUNC_EXCEPT:
        func_val = _eval(expr.children[0], env, state)
        key_val = _eval(expr.children[1], env, state)
        new_val = _eval(expr.children[2], env, state)
        if isinstance(func_val, FunctionValue):
            return func_val.except_update(key_val, new_val)
        raise EvalError(f"EXCEPT requires a function, got {type(func_val).__name__}")

    # --- DOMAIN -----------------------------------------------------------
    if kind is ExprKind.DOMAIN_OP:
        val = _eval(expr.children[0], env, state)
        if isinstance(val, FunctionValue):
            return val.domain()
        if isinstance(val, RecordValue):
            return SetValue(StringValue(n) for n in val.field_names())
        if isinstance(val, SequenceValue):
            if val.length() == 0:
                return SetValue()
            return SetValue(IntValue(i) for i in range(1, val.length() + 1))
        raise EvalError(f"DOMAIN requires function/record/sequence, got {type(val).__name__}")

    # --- tuple construction <<a, b, c>> -----------------------------------
    if kind is ExprKind.TUPLE_CONSTRUCT:
        elems = tuple(_eval(c, env, state) for c in expr.children)
        return TupleValue(elems)

    # --- record construction [a |-> 1, b |-> 2] --------------------------
    if kind is ExprKind.RECORD_CONSTRUCT:
        fields_def: Tuple[Tuple[str, Expr], ...] = expr.data
        fields = {name: _eval(body, env, state) for name, body in fields_def}
        return RecordValue(fields)

    # --- record field access r.field --------------------------------------
    if kind is ExprKind.RECORD_ACCESS:
        rec_val = _eval(expr.children[0], env, state)
        if not isinstance(rec_val, RecordValue):
            raise EvalError(f"Record access requires a record, got {type(rec_val).__name__}")
        return rec_val.access(expr.data)

    # --- record EXCEPT [r EXCEPT !.field = val] ---------------------------
    if kind is ExprKind.RECORD_EXCEPT:
        rec_val = _eval(expr.children[0], env, state)
        new_val = _eval(expr.children[1], env, state)
        if not isinstance(rec_val, RecordValue):
            raise EvalError(f"Record EXCEPT requires a record, got {type(rec_val).__name__}")
        return rec_val.except_update(expr.data, new_val)

    # --- sequence operations ----------------------------------------------
    if kind is ExprKind.SEQ_OP:
        return _eval_seq_op(expr.data, expr.children, env, state)

    # --- user-defined operator application --------------------------------
    if kind is ExprKind.OP_APPLY:
        return _eval_op_apply(expr.data, expr.children, env, state)

    # --- built-in module call ---------------------------------------------
    if kind is ExprKind.BUILTIN_CALL:
        module_name, op_name = expr.data
        return _eval_builtin_call(module_name, op_name, expr.children, env, state)

    # --- UNCHANGED --------------------------------------------------------
    if kind is ExprKind.UNCHANGED:
        # In expression context, UNCHANGED evaluates to TRUE iff for all
        # listed variables, the primed value equals the current value.
        var_names: Tuple[str, ...] = expr.data
        for vname in var_names:
            cur = state.get(vname)
            primed_key = vname + "'"
            primed = env.lookup(primed_key)
            if primed is None:
                raise EvalError(f"Primed variable '{vname}' not bound for UNCHANGED")
            if cur != primed:
                return BoolValue(False)
        return BoolValue(True)

    raise EvalError(f"Unknown expression kind: {kind}")


# ===================================================================
# Unary operators
# ===================================================================

def _eval_unary(op: str, operand_expr: Expr, env: Environment, state: TLAState) -> TLAValue:
    if op == "~" or op == "\\lnot" or op == "\\neg":
        return BoolValue(not _eval_bool(operand_expr, env, state, "~ operand"))

    if op == "-" or op == "MINUS":
        val = _eval(operand_expr, env, state)
        if not isinstance(val, IntValue):
            raise EvalError(f"Unary minus requires integer, got {type(val).__name__}")
        return IntValue(-val.val)

    if op == "SUBSET":
        val = _eval(operand_expr, env, state)
        if not isinstance(val, SetValue):
            raise EvalError(f"SUBSET requires a set, got {type(val).__name__}")
        return val.powerset()

    if op == "UNION":
        val = _eval(operand_expr, env, state)
        if not isinstance(val, SetValue):
            raise EvalError(f"UNION requires a set of sets, got {type(val).__name__}")
        return val.big_union()

    raise EvalError(f"Unknown unary operator: {op}")


# ===================================================================
# Binary operators
# ===================================================================

def _eval_binary(op: str, left_expr: Expr, right_expr: Expr,
                 env: Environment, state: TLAState) -> TLAValue:

    # --- short-circuit boolean operators ----------------------------------
    if op in ("/\\", "\\land", "&&"):
        lv = _eval_bool(left_expr, env, state, "/\\ left")
        if not lv:
            return BoolValue(False)
        return BoolValue(_eval_bool(right_expr, env, state, "/\\ right"))

    if op in ("\\/", "\\lor", "||"):
        lv = _eval_bool(left_expr, env, state, "\\/ left")
        if lv:
            return BoolValue(True)
        return BoolValue(_eval_bool(right_expr, env, state, "\\/ right"))

    if op in ("=>", "\\implies"):
        lv = _eval_bool(left_expr, env, state, "=> left")
        if not lv:
            return BoolValue(True)
        return BoolValue(_eval_bool(right_expr, env, state, "=> right"))

    if op in ("<=>", "\\equiv"):
        lv = _eval_bool(left_expr, env, state, "<=> left")
        rv = _eval_bool(right_expr, env, state, "<=> right")
        return BoolValue(lv == rv)

    # --- evaluate both sides for remaining operators ----------------------
    left = _eval(left_expr, env, state)
    right = _eval(right_expr, env, state)

    # --- arithmetic -------------------------------------------------------
    if op == "+":
        return IntValue(_as_int(left, "+") + _as_int(right, "+"))
    if op == "-":
        return IntValue(_as_int(left, "-") - _as_int(right, "-"))
    if op == "*":
        return IntValue(_as_int(left, "*") * _as_int(right, "*"))
    if op in ("\\div", "÷"):
        rv = _as_int(right, "\\div")
        if rv == 0:
            raise EvalError("Division by zero")
        lv = _as_int(left, "\\div")
        # TLA+ integer division truncates toward zero
        if (lv < 0) != (rv < 0) and lv % rv != 0:
            return IntValue(lv // rv + 1)
        return IntValue(lv // rv)
    if op == "%":
        rv = _as_int(right, "%")
        if rv == 0:
            raise EvalError("Modulo by zero")
        return IntValue(_as_int(left, "%") % rv)
    if op == "..":
        lo = _as_int(left, "..")
        hi = _as_int(right, "..")
        return SetValue(IntValue(i) for i in range(lo, hi + 1))
    if op == "^":
        base = _as_int(left, "^")
        exp = _as_int(right, "^")
        if exp < 0:
            raise EvalError("Negative exponent")
        return IntValue(base ** exp)

    # --- comparison -------------------------------------------------------
    if op == "=":
        return BoolValue(left == right)
    if op in ("/=", "#"):
        return BoolValue(left != right)
    if op == "<":
        return BoolValue(_as_int(left, "<") < _as_int(right, "<"))
    if op == ">":
        return BoolValue(_as_int(left, ">") > _as_int(right, ">"))
    if op == "<=":
        return BoolValue(_as_int(left, "<=") <= _as_int(right, "<="))
    if op in (">=", "\\geq"):
        return BoolValue(_as_int(left, ">=") >= _as_int(right, ">="))

    # --- set operations ---------------------------------------------------
    if op in ("\\in", "∈"):
        if not isinstance(right, SetValue):
            raise EvalError(f"\\in requires a set on the right, got {type(right).__name__}")
        return BoolValue(right.contains(left))

    if op in ("\\notin",):
        if not isinstance(right, SetValue):
            raise EvalError(f"\\notin requires a set, got {type(right).__name__}")
        return BoolValue(not right.contains(left))

    if op in ("\\union", "\\cup", "∪"):
        return _as_set(left, "\\union").union(_as_set(right, "\\union"))

    if op in ("\\intersect", "\\cap", "∩"):
        return _as_set(left, "\\intersect").intersect(_as_set(right, "\\intersect"))

    if op == "\\":
        return _as_set(left, "\\").difference(_as_set(right, "\\"))

    if op in ("\\subseteq", "⊆"):
        return BoolValue(_as_set(left, "\\subseteq").is_subset(_as_set(right, "\\subseteq")))

    if op in ("\\times", "\\X"):
        return _as_set(left, "\\times").cross(_as_set(right, "\\times"))

    # --- function/record operators ----------------------------------------
    if op == ":>":
        return FunctionValue(pairs=[(left, right)])

    if op == "@@":
        if isinstance(left, FunctionValue) and isinstance(right, FunctionValue):
            merged = dict(right.mapping)
            merged.update(left.mapping)
            return FunctionValue(merged)
        raise EvalError(f"@@ requires functions, got {type(left).__name__} and {type(right).__name__}")

    raise EvalError(f"Unknown binary operator: {op}")


# ===================================================================
# Compound expression helpers
# ===================================================================

def _eval_case(data: dict, env: Environment, state: TLAState) -> TLAValue:
    """Evaluate CASE cond1 -> e1 [] cond2 -> e2 [] ... [] OTHER -> eN."""
    arms: Tuple[Tuple[Expr, Expr], ...] = data["arms"]
    other: Optional[Expr] = data["other"]

    for cond_expr, result_expr in arms:
        cond = _eval_bool(cond_expr, env, state, "CASE condition")
        if cond:
            return _eval(result_expr, env, state)

    if other is not None:
        return _eval(other, env, state)
    raise EvalError("CASE fell through with no matching arm and no OTHER")


def _eval_forall(var: str, domain_expr: Expr, body_expr: Expr,
                 env: Environment, state: TLAState) -> TLAValue:
    """\\A var \\in S : P(var)."""
    domain = _as_set(_eval(domain_expr, env, state), "\\A domain")
    with env.scope("forall"):
        for elem in domain:
            env.bind(var, elem)
            if not _eval_bool(body_expr, env, state, f"\\A {var}"):
                return BoolValue(False)
    return BoolValue(True)


def _eval_exists(var: str, domain_expr: Expr, body_expr: Expr,
                 env: Environment, state: TLAState) -> TLAValue:
    """\\E var \\in S : P(var)."""
    domain = _as_set(_eval(domain_expr, env, state), "\\E domain")
    with env.scope("exists"):
        for elem in domain:
            env.bind(var, elem)
            if _eval_bool(body_expr, env, state, f"\\E {var}"):
                return BoolValue(True)
    return BoolValue(False)


def _eval_choose(var: str, domain_expr: Expr, pred_expr: Expr,
                 env: Environment, state: TLAState) -> TLAValue:
    """CHOOSE var \\in S : P(var) – pick the first satisfying element."""
    domain = _as_set(_eval(domain_expr, env, state), "CHOOSE domain")
    with env.scope("choose"):
        for elem in domain:
            env.bind(var, elem)
            if _eval_bool(pred_expr, env, state, f"CHOOSE {var}"):
                return elem
    raise EvalError(f"CHOOSE found no satisfying element in {domain.pretty()}")


def _eval_set_comp(var: str, domain_expr: Expr, map_expr: Expr,
                   env: Environment, state: TLAState) -> TLAValue:
    """{e(x) : x \\in S}."""
    domain = _as_set(_eval(domain_expr, env, state), "set comprehension domain")
    results: List[TLAValue] = []
    with env.scope("set_comp"):
        for elem in domain:
            env.bind(var, elem)
            results.append(_eval(map_expr, env, state))
    return SetValue(results)


def _eval_set_filter(var: str, domain_expr: Expr, pred_expr: Expr,
                     env: Environment, state: TLAState) -> TLAValue:
    """{x \\in S : P(x)}."""
    domain = _as_set(_eval(domain_expr, env, state), "set filter domain")
    results: List[TLAValue] = []
    with env.scope("set_filter"):
        for elem in domain:
            env.bind(var, elem)
            if _eval_bool(pred_expr, env, state, f"filter {var}"):
                results.append(elem)
    return SetValue(results)


def _eval_func_construct(var: str, domain_expr: Expr, body_expr: Expr,
                         env: Environment, state: TLAState) -> TLAValue:
    """[x \\in S |-> e(x)]."""
    domain = _as_set(_eval(domain_expr, env, state), "func construct domain")
    mapping: Dict[TLAValue, TLAValue] = {}
    with env.scope("func_construct"):
        for elem in domain:
            env.bind(var, elem)
            mapping[elem] = _eval(body_expr, env, state)
    return FunctionValue(mapping)


def _apply_function(func_val: TLAValue, arg_val: TLAValue) -> TLAValue:
    """Apply a function, sequence, or tuple to an argument."""
    if isinstance(func_val, FunctionValue):
        return func_val.apply(arg_val)
    if isinstance(func_val, SequenceValue):
        if not isinstance(arg_val, IntValue):
            raise EvalError(f"Sequence index must be an integer, got {type(arg_val).__name__}")
        return func_val.index(arg_val.val)
    if isinstance(func_val, TupleValue):
        if not isinstance(arg_val, IntValue):
            raise EvalError(f"Tuple index must be an integer, got {type(arg_val).__name__}")
        return func_val.index(arg_val.val)
    if isinstance(func_val, RecordValue):
        if not isinstance(arg_val, StringValue):
            raise EvalError(f"Record key must be a string, got {type(arg_val).__name__}")
        return func_val.access(arg_val.val)
    raise EvalError(f"Cannot apply {type(func_val).__name__} as a function")


# ===================================================================
# Sequence operations
# ===================================================================

def _eval_seq_op(op_name: str, children: Tuple[Expr, ...],
                 env: Environment, state: TLAState) -> TLAValue:
    if op_name == "Len":
        seq = _as_seq(_eval(children[0], env, state), "Len")
        return IntValue(seq.length())

    if op_name == "Head":
        seq = _as_seq(_eval(children[0], env, state), "Head")
        return seq.head()

    if op_name == "Tail":
        seq = _as_seq(_eval(children[0], env, state), "Tail")
        return seq.tail()

    if op_name == "Append":
        seq = _as_seq(_eval(children[0], env, state), "Append")
        elem = _eval(children[1], env, state)
        return seq.append(elem)

    if op_name == "Concat" or op_name == "\\o":
        s1 = _as_seq(_eval(children[0], env, state), "Concat left")
        s2 = _as_seq(_eval(children[1], env, state), "Concat right")
        return s1.concat(s2)

    if op_name == "SubSeq":
        seq = _as_seq(_eval(children[0], env, state), "SubSeq")
        m = _as_int(_eval(children[1], env, state), "SubSeq start")
        n = _as_int(_eval(children[2], env, state), "SubSeq end")
        return seq.sub_seq(m, n)

    if op_name == "SelectSeq":
        seq = _as_seq(_eval(children[0], env, state), "SelectSeq")
        # The predicate is an expression; we create a closure
        pred_expr = children[1]
        # SelectSeq(s, Test) where Test is a unary operator
        # We evaluate by applying the operator to each element
        def pred(elem: TLAValue) -> bool:
            with env.scope("selectseq"):
                env.bind("__selectseq_elem__", elem)
                result = _eval(pred_expr, env, state)
                if not isinstance(result, BoolValue):
                    raise EvalError("SelectSeq predicate must return BOOLEAN")
                return result.val
        return seq.select_seq(pred)

    raise EvalError(f"Unknown sequence operation: {op_name}")


# ===================================================================
# User-defined operator application
# ===================================================================

def _eval_op_apply(name: str, arg_exprs: Tuple[Expr, ...],
                   env: Environment, state: TLAState) -> TLAValue:
    opdef = env.get_operator(name)
    if opdef is None:
        # Try as a builtin
        builtin = env.get_builtin(name)
        if builtin is not None:
            return _call_builtin(builtin, arg_exprs, env, state)
        raise EvalError(f"Unknown operator '{name}'")

    if len(arg_exprs) != len(opdef.params):
        raise EvalError(
            f"Operator '{name}' expects {len(opdef.params)} args, got {len(arg_exprs)}"
        )

    arg_vals = [_eval(a, env, state) for a in arg_exprs]

    with env.scope(f"op:{name}"):
        for param, val in zip(opdef.params, arg_vals):
            env.bind(param, val)
        if opdef.body is None:
            raise EvalError(f"Operator '{name}' has no body")
        return _eval(opdef.body, env, state)


def _eval_builtin_call(module: str, name: str, arg_exprs: Tuple[Expr, ...],
                       env: Environment, state: TLAState) -> TLAValue:
    entry = env.get_module_builtin(module, name)
    if entry is None:
        entry = env.get_builtin(name)
    if entry is None:
        raise EvalError(f"Unknown builtin {module}!{name}")
    return _call_builtin(entry, arg_exprs, env, state)


def _call_builtin(entry, arg_exprs: Tuple[Expr, ...],
                  env: Environment, state: TLAState) -> TLAValue:
    if entry.is_lazy:
        return entry.evaluator(arg_exprs, env, state)
    arg_vals = [_eval(a, env, state) for a in arg_exprs]
    return entry.evaluator(*arg_vals)


# ===================================================================
# Type coercion helpers
# ===================================================================

def _as_int(val: TLAValue, context: str) -> int:
    if not isinstance(val, IntValue):
        raise EvalError(f"{context}: expected integer, got {type(val).__name__} ({val.pretty()})")
    return val.val

def _as_bool(val: TLAValue, context: str) -> bool:
    if not isinstance(val, BoolValue):
        raise EvalError(f"{context}: expected boolean, got {type(val).__name__} ({val.pretty()})")
    return val.val

def _eval_bool(expr: Expr, env: Environment, state: TLAState, context: str) -> bool:
    val = _eval(expr, env, state)
    return _as_bool(val, context)

def _as_set(val: TLAValue, context: str) -> SetValue:
    if not isinstance(val, SetValue):
        raise EvalError(f"{context}: expected set, got {type(val).__name__} ({val.pretty()})")
    return val

def _as_seq(val: TLAValue, context: str) -> SequenceValue:
    if isinstance(val, SequenceValue):
        return val
    if isinstance(val, TupleValue):
        return SequenceValue(val.elements)
    raise EvalError(f"{context}: expected sequence, got {type(val).__name__} ({val.pretty()})")
