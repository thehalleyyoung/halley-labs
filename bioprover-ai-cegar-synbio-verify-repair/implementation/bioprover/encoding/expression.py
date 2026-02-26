"""Expression IR for SMT encoding of biological system verification.

Provides an immutable, hashable expression tree with structural sharing
(hash-consing), operator overloading, and domain-specific nodes for
Hill functions, quantifiers, and Boolean connectives.
"""

from __future__ import annotations

import math
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Hash-consing table
# ---------------------------------------------------------------------------

_CONS_TABLE: Dict[Tuple, weakref.ref["ExprNode"]] = {}


def _intern(node: "ExprNode") -> "ExprNode":
    """Return the canonical representative for *node* (structural sharing)."""
    key = node._cons_key()
    ref = _CONS_TABLE.get(key)
    if ref is not None:
        existing = ref()
        if existing is not None:
            return existing
    _CONS_TABLE[key] = weakref.ref(node, lambda r, k=key: _CONS_TABLE.pop(k, None))
    return node


# ---------------------------------------------------------------------------
# Domain for quantified variables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """Closed interval [lo, hi]."""
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"Empty interval: [{self.lo}, {self.hi}]")

    def width(self) -> float:
        return self.hi - self.lo

    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2.0

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def subdivide(self, n: int = 2) -> List["Interval"]:
        step = self.width() / n
        return [Interval(self.lo + i * step, self.lo + (i + 1) * step) for i in range(n)]


# ---------------------------------------------------------------------------
# ExprNode base class
# ---------------------------------------------------------------------------

class ExprNode(ABC):
    """Immutable, hashable expression node with structural sharing."""

    __slots__ = ("_hash",)

    def __init__(self) -> None:
        object.__setattr__(self, "_hash", None)

    # -- structural identity ------------------------------------------------

    @abstractmethod
    def _cons_key(self) -> Tuple:
        """Return a tuple that uniquely identifies this node's structure."""

    def __hash__(self) -> int:
        h = object.__getattribute__(self, "_hash")
        if h is None:
            h = hash(self._cons_key())
            object.__setattr__(self, "_hash", h)
        return h

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, ExprNode):
            return NotImplemented
        return self._cons_key() == other._cons_key()

    # -- immutability -------------------------------------------------------

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ExprNode instances are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("ExprNode instances are immutable")

    # -- operator overloading -----------------------------------------------

    def __add__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Add(self, _wrap(other))

    def __radd__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Add(_wrap(other), self)

    def __sub__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Add(self, Neg(_wrap(other)))

    def __rsub__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Add(_wrap(other), Neg(self))

    def __mul__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Mul(self, _wrap(other))

    def __rmul__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Mul(_wrap(other), self)

    def __truediv__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Div(self, _wrap(other))

    def __rtruediv__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Div(_wrap(other), self)

    def __pow__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Pow(self, _wrap(other))

    def __neg__(self) -> "ExprNode":
        return Neg(self)

    def __pos__(self) -> "ExprNode":
        return self

    def __abs__(self) -> "ExprNode":
        return Abs(self)

    # -- comparison constructors (return ExprNode, not bool) ----------------

    def __lt__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Lt(self, _wrap(other))

    def __le__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Le(self, _wrap(other))

    def __ge__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Ge(self, _wrap(other))

    def __gt__(self, other: Union["ExprNode", float, int]) -> "ExprNode":
        return Gt(self, _wrap(other))

    # -- traversal ----------------------------------------------------------

    @abstractmethod
    def children(self) -> Tuple["ExprNode", ...]:
        """Return direct children of this node."""

    def free_vars(self) -> FrozenSet[str]:
        """Collect all free variable names."""
        result: Set[str] = set()
        self._collect_free_vars(result, set())
        return frozenset(result)

    def _collect_free_vars(self, acc: Set[str], bound: Set[str]) -> None:
        for child in self.children():
            child._collect_free_vars(acc, bound)

    def depth(self) -> int:
        kids = self.children()
        if not kids:
            return 0
        return 1 + max(c.depth() for c in kids)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children())

    def substitute(self, mapping: Dict[str, "ExprNode"]) -> "ExprNode":
        """Substitute variables according to *mapping*."""
        return self._subst(mapping)

    @abstractmethod
    def _subst(self, mapping: Dict[str, "ExprNode"]) -> "ExprNode":
        ...

    # -- iteration ----------------------------------------------------------

    def iter_preorder(self) -> Iterator["ExprNode"]:
        yield self
        for c in self.children():
            yield from c.iter_preorder()

    def iter_postorder(self) -> Iterator["ExprNode"]:
        for c in self.children():
            yield from c.iter_postorder()
        yield self

    # -- pretty printing ----------------------------------------------------

    @abstractmethod
    def _pretty(self, prec: int) -> str:
        ...

    def pretty(self) -> str:
        return self._pretty(0)

    def __repr__(self) -> str:
        return self.pretty()


# ---------------------------------------------------------------------------
# Helper to wrap python scalars
# ---------------------------------------------------------------------------

def _wrap(v: Union[ExprNode, float, int]) -> ExprNode:
    if isinstance(v, ExprNode):
        return v
    if isinstance(v, (int, float)):
        return Const(float(v))
    raise TypeError(f"Cannot wrap {type(v)} as ExprNode")


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class Const(ExprNode):
    """Numeric constant."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        super().__init__()
        object.__setattr__(self, "value", value)

    def _cons_key(self) -> Tuple:
        return ("Const", self.value)

    def children(self) -> Tuple[ExprNode, ...]:
        return ()

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        return self

    def _pretty(self, prec: int) -> str:
        v = self.value
        if v == int(v) and not math.isinf(v):
            return str(int(v))
        return str(v)


class Var(ExprNode):
    """Named variable."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        super().__init__()
        object.__setattr__(self, "name", name)

    def _cons_key(self) -> Tuple:
        return ("Var", self.name)

    def children(self) -> Tuple[ExprNode, ...]:
        return ()

    def _collect_free_vars(self, acc: Set[str], bound: Set[str]) -> None:
        if self.name not in bound:
            acc.add(self.name)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        return mapping.get(self.name, self)

    def _pretty(self, prec: int) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------------------

class _BinOp(ExprNode):
    """Abstract binary operator."""

    __slots__ = ("left", "right")
    _op_str: str = "?"
    _prec: int = 0

    def __init__(self, left: ExprNode, right: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    def _cons_key(self) -> Tuple:
        return (type(self).__name__, id(self.left) if self.left is self.left else self.left._cons_key(), self.right._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.left, self.right)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        new_l = self.left._subst(mapping)
        new_r = self.right._subst(mapping)
        if new_l is self.left and new_r is self.right:
            return self
        return type(self)(new_l, new_r)

    def _pretty(self, prec: int) -> str:
        s = f"{self.left._pretty(self._prec)} {self._op_str} {self.right._pretty(self._prec + 1)}"
        if prec > self._prec:
            return f"({s})"
        return s

    def _cons_key(self) -> Tuple:
        return (type(self).__name__, self.left._cons_key(), self.right._cons_key())


class Add(_BinOp):
    _op_str = "+"
    _prec = 1


class Mul(_BinOp):
    _op_str = "*"
    _prec = 2


class Div(_BinOp):
    _op_str = "/"
    _prec = 2


class Pow(_BinOp):
    _op_str = "^"
    _prec = 3


# ---------------------------------------------------------------------------
# Unary nodes
# ---------------------------------------------------------------------------

class _UnaryOp(ExprNode):
    """Abstract unary operator."""

    __slots__ = ("operand",)
    _func_name: str = "?"

    def __init__(self, operand: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "operand", operand)

    def _cons_key(self) -> Tuple:
        return (type(self).__name__, self.operand._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.operand,)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        new_op = self.operand._subst(mapping)
        if new_op is self.operand:
            return self
        return type(self)(new_op)

    def _pretty(self, prec: int) -> str:
        return f"{self._func_name}({self.operand._pretty(0)})"


class Neg(_UnaryOp):
    _func_name = "-"

    def _pretty(self, prec: int) -> str:
        s = f"-{self.operand._pretty(4)}"
        if prec > 3:
            return f"({s})"
        return s


class Exp(_UnaryOp):
    _func_name = "exp"


class Log(_UnaryOp):
    _func_name = "log"


class Sin(_UnaryOp):
    _func_name = "sin"


class Cos(_UnaryOp):
    _func_name = "cos"


class Sqrt(_UnaryOp):
    _func_name = "sqrt"


class Abs(_UnaryOp):
    _func_name = "abs"


# ---------------------------------------------------------------------------
# Min / Max
# ---------------------------------------------------------------------------

class Min(_BinOp):
    _op_str = "min"
    _prec = 0

    def _pretty(self, prec: int) -> str:
        return f"min({self.left._pretty(0)}, {self.right._pretty(0)})"


class Max(_BinOp):
    _op_str = "max"
    _prec = 0

    def _pretty(self, prec: int) -> str:
        return f"max({self.left._pretty(0)}, {self.right._pretty(0)})"


# ---------------------------------------------------------------------------
# Hill function nodes
# ---------------------------------------------------------------------------

class HillAct(ExprNode):
    """Activating Hill function: x^n / (K^n + x^n)."""

    __slots__ = ("x", "K", "n")

    def __init__(self, x: ExprNode, K: ExprNode, n: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "n", n)

    def _cons_key(self) -> Tuple:
        return ("HillAct", self.x._cons_key(), self.K._cons_key(), self.n._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.x, self.K, self.n)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        nx = self.x._subst(mapping)
        nk = self.K._subst(mapping)
        nn = self.n._subst(mapping)
        if nx is self.x and nk is self.K and nn is self.n:
            return self
        return HillAct(nx, nk, nn)

    def _pretty(self, prec: int) -> str:
        return f"HillAct({self.x._pretty(0)}, {self.K._pretty(0)}, {self.n._pretty(0)})"


class HillRep(ExprNode):
    """Repressing Hill function: K^n / (K^n + x^n)."""

    __slots__ = ("x", "K", "n")

    def __init__(self, x: ExprNode, K: ExprNode, n: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "n", n)

    def _cons_key(self) -> Tuple:
        return ("HillRep", self.x._cons_key(), self.K._cons_key(), self.n._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.x, self.K, self.n)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        nx = self.x._subst(mapping)
        nk = self.K._subst(mapping)
        nn = self.n._subst(mapping)
        if nx is self.x and nk is self.K and nn is self.n:
            return self
        return HillRep(nx, nk, nn)

    def _pretty(self, prec: int) -> str:
        return f"HillRep({self.x._pretty(0)}, {self.K._pretty(0)}, {self.n._pretty(0)})"


# ---------------------------------------------------------------------------
# Comparison nodes
# ---------------------------------------------------------------------------

class _CmpOp(_BinOp):
    _prec = 0


class Lt(_CmpOp):
    _op_str = "<"


class Le(_CmpOp):
    _op_str = "<="


class Eq(_CmpOp):
    _op_str = "=="

    def __bool__(self) -> bool:
        raise TypeError("Eq node is symbolic; use structural equality via 'is' or ExprNode.__eq__")


class Ge(_CmpOp):
    _op_str = ">="


class Gt(_CmpOp):
    _op_str = ">"


# ---------------------------------------------------------------------------
# Boolean connective nodes
# ---------------------------------------------------------------------------

class And(ExprNode):
    """Logical conjunction of two or more sub-expressions."""

    __slots__ = ("args",)

    def __init__(self, *args: ExprNode) -> None:
        super().__init__()
        flat: List[ExprNode] = []
        for a in args:
            if isinstance(a, And):
                flat.extend(a.args)
            else:
                flat.append(a)
        object.__setattr__(self, "args", tuple(flat))

    def _cons_key(self) -> Tuple:
        return ("And",) + tuple(a._cons_key() for a in self.args)

    def children(self) -> Tuple[ExprNode, ...]:
        return self.args

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        new_args = tuple(a._subst(mapping) for a in self.args)
        if all(n is o for n, o in zip(new_args, self.args)):
            return self
        return And(*new_args)

    def _pretty(self, prec: int) -> str:
        inner = " ∧ ".join(a._pretty(0) for a in self.args)
        if prec > 0:
            return f"({inner})"
        return inner


class Or(ExprNode):
    """Logical disjunction of two or more sub-expressions."""

    __slots__ = ("args",)

    def __init__(self, *args: ExprNode) -> None:
        super().__init__()
        flat: List[ExprNode] = []
        for a in args:
            if isinstance(a, Or):
                flat.extend(a.args)
            else:
                flat.append(a)
        object.__setattr__(self, "args", tuple(flat))

    def _cons_key(self) -> Tuple:
        return ("Or",) + tuple(a._cons_key() for a in self.args)

    def children(self) -> Tuple[ExprNode, ...]:
        return self.args

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        new_args = tuple(a._subst(mapping) for a in self.args)
        if all(n is o for n, o in zip(new_args, self.args)):
            return self
        return Or(*new_args)

    def _pretty(self, prec: int) -> str:
        inner = " ∨ ".join(a._pretty(0) for a in self.args)
        if prec > 0:
            return f"({inner})"
        return inner


class Not(ExprNode):
    """Logical negation."""

    __slots__ = ("operand",)

    def __init__(self, operand: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "operand", operand)

    def _cons_key(self) -> Tuple:
        return ("Not", self.operand._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.operand,)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        new_op = self.operand._subst(mapping)
        if new_op is self.operand:
            return self
        return Not(new_op)

    def _pretty(self, prec: int) -> str:
        return f"¬{self.operand._pretty(5)}"


class Implies(ExprNode):
    """Logical implication: antecedent → consequent."""

    __slots__ = ("antecedent", "consequent")

    def __init__(self, antecedent: ExprNode, consequent: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "antecedent", antecedent)
        object.__setattr__(self, "consequent", consequent)

    def _cons_key(self) -> Tuple:
        return ("Implies", self.antecedent._cons_key(), self.consequent._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.antecedent, self.consequent)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        new_a = self.antecedent._subst(mapping)
        new_c = self.consequent._subst(mapping)
        if new_a is self.antecedent and new_c is self.consequent:
            return self
        return Implies(new_a, new_c)

    def _pretty(self, prec: int) -> str:
        s = f"{self.antecedent._pretty(1)} → {self.consequent._pretty(0)}"
        if prec > 0:
            return f"({s})"
        return s


class Ite(ExprNode):
    """If-then-else: ite(cond, then_expr, else_expr)."""

    __slots__ = ("cond", "then_expr", "else_expr")

    def __init__(self, cond: ExprNode, then_expr: ExprNode, else_expr: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "cond", cond)
        object.__setattr__(self, "then_expr", then_expr)
        object.__setattr__(self, "else_expr", else_expr)

    def _cons_key(self) -> Tuple:
        return ("Ite", self.cond._cons_key(), self.then_expr._cons_key(), self.else_expr._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.cond, self.then_expr, self.else_expr)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        nc = self.cond._subst(mapping)
        nt = self.then_expr._subst(mapping)
        ne = self.else_expr._subst(mapping)
        if nc is self.cond and nt is self.then_expr and ne is self.else_expr:
            return self
        return Ite(nc, nt, ne)

    def _pretty(self, prec: int) -> str:
        return f"ite({self.cond._pretty(0)}, {self.then_expr._pretty(0)}, {self.else_expr._pretty(0)})"


# ---------------------------------------------------------------------------
# Quantifier nodes
# ---------------------------------------------------------------------------

class ForAll(ExprNode):
    """Universal quantification: ∀ var ∈ domain. body."""

    __slots__ = ("var", "domain", "body")

    def __init__(self, var: str, domain: Optional[Interval], body: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "var", var)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "body", body)

    def _cons_key(self) -> Tuple:
        d = (self.domain.lo, self.domain.hi) if self.domain else None
        return ("ForAll", self.var, d, self.body._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.body,)

    def _collect_free_vars(self, acc: Set[str], bound: Set[str]) -> None:
        new_bound = bound | {self.var}
        self.body._collect_free_vars(acc, new_bound)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        safe = {k: v for k, v in mapping.items() if k != self.var}
        new_body = self.body._subst(safe)
        if new_body is self.body:
            return self
        return ForAll(self.var, self.domain, new_body)

    def _pretty(self, prec: int) -> str:
        dom = f" ∈ [{self.domain.lo}, {self.domain.hi}]" if self.domain else ""
        return f"∀{self.var}{dom}. {self.body._pretty(0)}"


class Exists(ExprNode):
    """Existential quantification: ∃ var ∈ domain. body."""

    __slots__ = ("var", "domain", "body")

    def __init__(self, var: str, domain: Optional[Interval], body: ExprNode) -> None:
        super().__init__()
        object.__setattr__(self, "var", var)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "body", body)

    def _cons_key(self) -> Tuple:
        d = (self.domain.lo, self.domain.hi) if self.domain else None
        return ("Exists", self.var, d, self.body._cons_key())

    def children(self) -> Tuple[ExprNode, ...]:
        return (self.body,)

    def _collect_free_vars(self, acc: Set[str], bound: Set[str]) -> None:
        new_bound = bound | {self.var}
        self.body._collect_free_vars(acc, new_bound)

    def _subst(self, mapping: Dict[str, ExprNode]) -> ExprNode:
        safe = {k: v for k, v in mapping.items() if k != self.var}
        new_body = self.body._subst(safe)
        if new_body is self.body:
            return self
        return Exists(self.var, self.domain, new_body)

    def _pretty(self, prec: int) -> str:
        dom = f" ∈ [{self.domain.lo}, {self.domain.hi}]" if self.domain else ""
        return f"∃{self.var}{dom}. {self.body._pretty(0)}"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def const(v: float) -> Const:
    return Const(v)


def var(name: str) -> Var:
    return Var(name)


ZERO = Const(0.0)
ONE = Const(1.0)
TWO = Const(2.0)


def sum_exprs(exprs: Sequence[ExprNode]) -> ExprNode:
    """Sum a sequence of expressions, returning ZERO for empty."""
    if not exprs:
        return ZERO
    result = exprs[0]
    for e in exprs[1:]:
        result = Add(result, e)
    return result


def prod_exprs(exprs: Sequence[ExprNode]) -> ExprNode:
    """Product of a sequence of expressions, returning ONE for empty."""
    if not exprs:
        return ONE
    result = exprs[0]
    for e in exprs[1:]:
        result = Mul(result, e)
    return result


# ---------------------------------------------------------------------------
# Expression map / transform
# ---------------------------------------------------------------------------

def map_expr(e: ExprNode, fn: Callable[[ExprNode], Optional[ExprNode]]) -> ExprNode:
    """Bottom-up rewrite: apply *fn* to every node. If *fn* returns None the
    node is kept as-is (after its children have been rewritten)."""
    # rewrite children first
    if isinstance(e, (Const, Var)):
        new_e = e
    elif isinstance(e, _UnaryOp):
        new_child = map_expr(e.operand, fn)
        new_e = type(e)(new_child) if new_child is not e.operand else e
    elif isinstance(e, _BinOp):
        new_l = map_expr(e.left, fn)
        new_r = map_expr(e.right, fn)
        new_e = type(e)(new_l, new_r) if (new_l is not e.left or new_r is not e.right) else e
    elif isinstance(e, (And, Or)):
        new_args = tuple(map_expr(a, fn) for a in e.args)
        changed = any(n is not o for n, o in zip(new_args, e.args))
        new_e = type(e)(*new_args) if changed else e
    elif isinstance(e, Not):
        new_op = map_expr(e.operand, fn)
        new_e = Not(new_op) if new_op is not e.operand else e
    elif isinstance(e, Implies):
        na = map_expr(e.antecedent, fn)
        nc = map_expr(e.consequent, fn)
        new_e = Implies(na, nc) if (na is not e.antecedent or nc is not e.consequent) else e
    elif isinstance(e, Ite):
        nc = map_expr(e.cond, fn)
        nt = map_expr(e.then_expr, fn)
        ne = map_expr(e.else_expr, fn)
        new_e = Ite(nc, nt, ne) if (nc is not e.cond or nt is not e.then_expr or ne is not e.else_expr) else e
    elif isinstance(e, (HillAct, HillRep)):
        nx = map_expr(e.x, fn)
        nk = map_expr(e.K, fn)
        nn = map_expr(e.n, fn)
        new_e = type(e)(nx, nk, nn) if (nx is not e.x or nk is not e.K or nn is not e.n) else e
    elif isinstance(e, (ForAll, Exists)):
        nb = map_expr(e.body, fn)
        new_e = type(e)(e.var, e.domain, nb) if nb is not e.body else e
    else:
        new_e = e

    result = fn(new_e)
    return result if result is not None else new_e


def collect_nodes(e: ExprNode, pred: Callable[[ExprNode], bool]) -> List[ExprNode]:
    """Collect all nodes satisfying *pred* in pre-order."""
    result: List[ExprNode] = []
    for node in e.iter_preorder():
        if pred(node):
            result.append(node)
    return result


def is_const(e: ExprNode) -> bool:
    return isinstance(e, Const)


def is_var(e: ExprNode) -> bool:
    return isinstance(e, Var)


def const_value(e: ExprNode) -> Optional[float]:
    """Return the constant value if *e* is a Const, else None."""
    if isinstance(e, Const):
        return e.value
    return None
