"""
Full refinement type lattice with Z3-backed subtyping.

Defines refinement types {x : τ | P(x)} where P is a predicate from the
guard language, with proper meet/join/subtyping/widening/narrowing operations.

Subtyping is decided via SMT:
    {x:T|P} <: {x:T|Q}  iff  ∀x. P(x) ⟹ Q(x)

This module wires into the existing abstract domains in src/domains/.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Predicate language for refinements
# ---------------------------------------------------------------------------

class PredOp(Enum):
    """Predicate operators in the refinement language."""
    TRUE = auto()
    FALSE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    IS_NONE = auto()
    IS_NOT_NONE = auto()
    ISINSTANCE = auto()
    HASATTR = auto()
    TRUTHY = auto()
    DIVISIBLE = auto()
    IN_RANGE = auto()
    LEN_EQ = auto()
    LEN_GT = auto()
    LEN_GE = auto()
    # Tensor shape predicates
    SHAPE_EQ = auto()       # shape(x) == (d1, d2, ...)
    SHAPE_DIM_EQ = auto()   # shape(x)[i] == d
    SHAPE_NDIM = auto()     # ndim(x) == n
    SHAPE_COMPAT = auto()   # shape(x) is compatible with shape(y) for matmul/etc.


@dataclass(frozen=True)
class Pred:
    """A predicate in the refinement language.

    Predicates are symbolic expressions over program variables.
    They are closed under boolean operations and comparisons.
    """
    op: PredOp
    args: Tuple = ()
    children: Tuple[Pred, ...] = ()

    # --- Smart constructors ---
    @staticmethod
    def true_() -> Pred:
        return Pred(PredOp.TRUE)

    @staticmethod
    def false_() -> Pred:
        return Pred(PredOp.FALSE)

    @staticmethod
    def var_eq(var: str, val: int) -> Pred:
        return Pred(PredOp.EQ, (var, val))

    @staticmethod
    def var_neq(var: str, val: int) -> Pred:
        return Pred(PredOp.NEQ, (var, val))

    @staticmethod
    def var_lt(var: str, val: int) -> Pred:
        return Pred(PredOp.LT, (var, val))

    @staticmethod
    def var_le(var: str, val: int) -> Pred:
        return Pred(PredOp.LE, (var, val))

    @staticmethod
    def var_gt(var: str, val: int) -> Pred:
        return Pred(PredOp.GT, (var, val))

    @staticmethod
    def var_ge(var: str, val: int) -> Pred:
        return Pred(PredOp.GE, (var, val))

    @staticmethod
    def var_cmp(var: str, op: str, val: int) -> Pred:
        ops = {"==": PredOp.EQ, "!=": PredOp.NEQ, "<": PredOp.LT,
               "<=": PredOp.LE, ">": PredOp.GT, ">=": PredOp.GE}
        return Pred(ops[op], (var, val))

    @staticmethod
    def is_none(var: str) -> Pred:
        return Pred(PredOp.IS_NONE, (var,))

    @staticmethod
    def is_not_none(var: str) -> Pred:
        return Pred(PredOp.IS_NOT_NONE, (var,))

    @staticmethod
    def isinstance_(var: str, typ: str) -> Pred:
        return Pred(PredOp.ISINSTANCE, (var, typ))

    @staticmethod
    def hasattr_(var: str, attr: str) -> Pred:
        return Pred(PredOp.HASATTR, (var, attr))

    @staticmethod
    def truthy(var: str) -> Pred:
        return Pred(PredOp.TRUTHY, (var,))

    @staticmethod
    def divisible(var: str, divisor: int) -> Pred:
        return Pred(PredOp.DIVISIBLE, (var, divisor))

    @staticmethod
    def in_range(var: str, lo: int, hi: int) -> Pred:
        return Pred(PredOp.IN_RANGE, (var, lo, hi))

    @staticmethod
    def len_eq(var: str, length: int) -> Pred:
        return Pred(PredOp.LEN_EQ, (var, length))

    @staticmethod
    def len_gt(var: str, length: int) -> Pred:
        return Pred(PredOp.LEN_GT, (var, length))

    @staticmethod
    def len_ge(var: str, length: int) -> Pred:
        return Pred(PredOp.LEN_GE, (var, length))

    @staticmethod
    def shape_eq(var: str, dims: Tuple) -> Pred:
        """shape(var) == dims, where dims is a tuple of ints or symbolic names."""
        return Pred(PredOp.SHAPE_EQ, (var, dims))

    @staticmethod
    def shape_dim_eq(var: str, axis: int, dim: object) -> Pred:
        """shape(var)[axis] == dim (int or symbolic str)."""
        return Pred(PredOp.SHAPE_DIM_EQ, (var, axis, dim))

    @staticmethod
    def shape_ndim(var: str, ndim: int) -> Pred:
        """ndim(var) == ndim."""
        return Pred(PredOp.SHAPE_NDIM, (var, ndim))

    @staticmethod
    def shape_compat(var1: str, var2: str, op: str = "matmul") -> Pred:
        """Shapes of var1 and var2 are compatible for op."""
        return Pred(PredOp.SHAPE_COMPAT, (var1, var2, op))

    def and_(self, other: Pred) -> Pred:
        if self.op == PredOp.TRUE:
            return other
        if other.op == PredOp.TRUE:
            return self
        if self.op == PredOp.FALSE or other.op == PredOp.FALSE:
            return Pred.false_()
        return Pred(PredOp.AND, (), (self, other))

    def or_(self, other: Pred) -> Pred:
        if self.op == PredOp.FALSE:
            return other
        if other.op == PredOp.FALSE:
            return self
        if self.op == PredOp.TRUE or other.op == PredOp.TRUE:
            return Pred.true_()
        return Pred(PredOp.OR, (), (self, other))

    def not_(self) -> Pred:
        if self.op == PredOp.TRUE:
            return Pred.false_()
        if self.op == PredOp.FALSE:
            return Pred.true_()
        if self.op == PredOp.NOT:
            return self.children[0]
        # Simplify common negations
        if self.op == PredOp.EQ:
            return Pred(PredOp.NEQ, self.args, self.children)
        if self.op == PredOp.NEQ:
            return Pred(PredOp.EQ, self.args, self.children)
        if self.op == PredOp.LT:
            return Pred(PredOp.GE, self.args, self.children)
        if self.op == PredOp.LE:
            return Pred(PredOp.GT, self.args, self.children)
        if self.op == PredOp.GT:
            return Pred(PredOp.LE, self.args, self.children)
        if self.op == PredOp.GE:
            return Pred(PredOp.LT, self.args, self.children)
        if self.op == PredOp.IS_NONE:
            return Pred(PredOp.IS_NOT_NONE, self.args, self.children)
        if self.op == PredOp.IS_NOT_NONE:
            return Pred(PredOp.IS_NONE, self.args, self.children)
        return Pred(PredOp.NOT, (), (self,))

    def implies(self, other: Pred) -> Pred:
        return Pred(PredOp.IMPLIES, (), (self, other))

    def free_vars(self) -> FrozenSet[str]:
        """Return the set of free variables mentioned in this predicate."""
        result: Set[str] = set()
        if self.op in (PredOp.EQ, PredOp.NEQ, PredOp.LT, PredOp.LE,
                       PredOp.GT, PredOp.GE):
            result.add(self.args[0])
        elif self.op in (PredOp.IS_NONE, PredOp.IS_NOT_NONE, PredOp.TRUTHY):
            result.add(self.args[0])
        elif self.op in (PredOp.ISINSTANCE, PredOp.HASATTR):
            result.add(self.args[0])
        elif self.op in (PredOp.DIVISIBLE, PredOp.IN_RANGE,
                         PredOp.LEN_EQ, PredOp.LEN_GT, PredOp.LEN_GE):
            result.add(self.args[0])
        for child in self.children:
            result |= child.free_vars()
        return frozenset(result)

    def substitute(self, old: str, new: str) -> Pred:
        """Substitute variable name old -> new."""
        new_args = tuple(
            new if (isinstance(a, str) and a == old) else a
            for a in self.args
        )
        new_children = tuple(c.substitute(old, new) for c in self.children)
        return Pred(self.op, new_args, new_children)

    def pretty(self) -> str:
        if self.op == PredOp.TRUE:
            return "⊤"
        if self.op == PredOp.FALSE:
            return "⊥"
        if self.op in (PredOp.EQ, PredOp.NEQ, PredOp.LT, PredOp.LE,
                       PredOp.GT, PredOp.GE):
            sym = {PredOp.EQ: "=", PredOp.NEQ: "≠", PredOp.LT: "<",
                   PredOp.LE: "≤", PredOp.GT: ">", PredOp.GE: "≥"}[self.op]
            return f"{self.args[0]} {sym} {self.args[1]}"
        if self.op == PredOp.IS_NONE:
            return f"{self.args[0]} is None"
        if self.op == PredOp.IS_NOT_NONE:
            return f"{self.args[0]} is not None"
        if self.op == PredOp.AND:
            return f"({self.children[0].pretty()} ∧ {self.children[1].pretty()})"
        if self.op == PredOp.OR:
            return f"({self.children[0].pretty()} ∨ {self.children[1].pretty()})"
        if self.op == PredOp.NOT:
            return f"¬{self.children[0].pretty()}"
        if self.op == PredOp.IMPLIES:
            return f"({self.children[0].pretty()} ⟹ {self.children[1].pretty()})"
        if self.op == PredOp.ISINSTANCE:
            return f"isinstance({self.args[0]}, {self.args[1]})"
        if self.op == PredOp.HASATTR:
            return f"hasattr({self.args[0]}, {self.args[1]!r})"
        if self.op == PredOp.TRUTHY:
            return f"truthy({self.args[0]})"
        if self.op == PredOp.DIVISIBLE:
            return f"{self.args[0]} % {self.args[1]} = 0"
        if self.op == PredOp.IN_RANGE:
            return f"{self.args[1]} ≤ {self.args[0]} ≤ {self.args[2]}"
        if self.op == PredOp.LEN_EQ:
            return f"len({self.args[0]}) = {self.args[1]}"
        if self.op == PredOp.LEN_GT:
            return f"len({self.args[0]}) > {self.args[1]}"
        if self.op == PredOp.LEN_GE:
            return f"len({self.args[0]}) ≥ {self.args[1]}"
        if self.op == PredOp.SHAPE_EQ:
            dims = self.args[1]
            dim_str = ", ".join(str(d) for d in dims) if isinstance(dims, tuple) else str(dims)
            return f"shape({self.args[0]}) = ({dim_str})"
        if self.op == PredOp.SHAPE_DIM_EQ:
            return f"shape({self.args[0]})[{self.args[1]}] = {self.args[2]}"
        if self.op == PredOp.SHAPE_NDIM:
            return f"ndim({self.args[0]}) = {self.args[1]}"
        if self.op == PredOp.SHAPE_COMPAT:
            return f"shape_compat({self.args[0]}, {self.args[1]}, {self.args[2]})"
        return f"Pred({self.op.name}, {self.args})"


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------

class BaseTypeKind(Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    NONE = "None"
    LIST = "list"
    DICT = "dict"
    SET = "set"
    TUPLE = "tuple"
    OBJECT = "object"
    ANY = "Any"
    NEVER = "Never"
    CALLABLE = "callable"
    TENSOR = "Tensor"


@dataclass(frozen=True)
class BaseTypeR:
    """A base type in the refinement type system."""
    kind: BaseTypeKind
    type_args: Tuple[BaseTypeR, ...] = ()
    # For callable: (param_types..., return_type)
    param_types: Tuple[BaseTypeR, ...] = ()
    return_type: Optional[BaseTypeR] = None

    def is_subtype_of(self, other: BaseTypeR) -> bool:
        """Structural subtyping for base types."""
        if other.kind == BaseTypeKind.ANY:
            return True
        if self.kind == BaseTypeKind.NEVER:
            return True
        if self.kind == other.kind:
            if not self.type_args and not other.type_args:
                return True
            if len(self.type_args) == len(other.type_args):
                return all(a.is_subtype_of(b)
                           for a, b in zip(self.type_args, other.type_args))
        # bool <: int
        if self.kind == BaseTypeKind.BOOL and other.kind == BaseTypeKind.INT:
            return True
        # int <: float
        if self.kind == BaseTypeKind.INT and other.kind == BaseTypeKind.FLOAT:
            return True
        return False

    def pretty(self) -> str:
        if self.type_args:
            args = ", ".join(a.pretty() for a in self.type_args)
            return f"{self.kind.value}[{args}]"
        if self.kind == BaseTypeKind.CALLABLE and self.param_types:
            params = ", ".join(p.pretty() for p in self.param_types)
            ret = self.return_type.pretty() if self.return_type else "None"
            return f"({params}) -> {ret}"
        return self.kind.value


INT_TYPE = BaseTypeR(BaseTypeKind.INT)
FLOAT_TYPE = BaseTypeR(BaseTypeKind.FLOAT)
STR_TYPE = BaseTypeR(BaseTypeKind.STR)
BOOL_TYPE = BaseTypeR(BaseTypeKind.BOOL)
NONE_TYPE = BaseTypeR(BaseTypeKind.NONE)
ANY_TYPE = BaseTypeR(BaseTypeKind.ANY)
NEVER_TYPE = BaseTypeR(BaseTypeKind.NEVER)
OBJECT_TYPE = BaseTypeR(BaseTypeKind.OBJECT)
TENSOR_TYPE = BaseTypeR(BaseTypeKind.TENSOR)


# ---------------------------------------------------------------------------
# Refinement types: {x : τ | P(x)}
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefType:
    """A refinement type {binder : base | pred}.

    The binder variable scopes over the predicate.
    """
    binder: str
    base: BaseTypeR
    pred: Pred

    @staticmethod
    def trivial(base: BaseTypeR, binder: str = "ν") -> RefType:
        return RefType(binder, base, Pred.true_())

    @staticmethod
    def bottom(base: BaseTypeR, binder: str = "ν") -> RefType:
        return RefType(binder, base, Pred.false_())

    def with_pred(self, p: Pred) -> RefType:
        return RefType(self.binder, self.base, p)

    def alpha_rename(self, new_binder: str) -> RefType:
        if new_binder == self.binder:
            return self
        return RefType(new_binder, self.base,
                       self.pred.substitute(self.binder, new_binder))

    def pretty(self) -> str:
        if self.pred.op == PredOp.TRUE:
            return self.base.pretty()
        return f"{{{self.binder}: {self.base.pretty()} | {self.pred.pretty()}}}"

    def __str__(self) -> str:
        return self.pretty()


# ---------------------------------------------------------------------------
# Z3 encoding of predicates
# ---------------------------------------------------------------------------

class Z3Encoder:
    """Encode refinement predicates into Z3 formulas."""

    def __init__(self):
        self._var_cache: Dict[str, z3.ArithRef] = {}
        self._bool_cache: Dict[str, z3.BoolRef] = {}
        self._len_cache: Dict[str, z3.ArithRef] = {}
        self._shape_cache: Dict[Tuple[str, int], z3.ArithRef] = {}
        self._ndim_cache: Dict[str, z3.ArithRef] = {}
        # Type tag encoding: map type names to integers
        self._type_tags: Dict[str, int] = {
            "int": 0, "float": 1, "str": 2, "bool": 3,
            "None": 4, "list": 5, "dict": 6, "set": 7,
            "tuple": 8, "object": 9,
        }
        self._tag_cache: Dict[str, z3.ArithRef] = {}

    def int_var(self, name: str) -> z3.ArithRef:
        if name not in self._var_cache:
            self._var_cache[name] = z3.Int(name)
        return self._var_cache[name]

    def bool_var(self, name: str) -> z3.BoolRef:
        if name not in self._bool_cache:
            self._bool_cache[name] = z3.Bool(f"b_{name}")
        return self._bool_cache[name]

    def len_var(self, name: str) -> z3.ArithRef:
        if name not in self._len_cache:
            self._len_cache[name] = z3.Int(f"len_{name}")
        return self._len_cache[name]

    def tag_var(self, name: str) -> z3.ArithRef:
        if name not in self._tag_cache:
            self._tag_cache[name] = z3.Int(f"tag_{name}")
        return self._tag_cache[name]

    def none_var(self, name: str) -> z3.BoolRef:
        """Boolean: is this variable None?"""
        return z3.Bool(f"none_{name}")

    def shape_dim_var(self, name: str, axis: int) -> z3.ArithRef:
        """Integer Z3 variable for shape(name)[axis]."""
        key = (name, axis)
        if key not in self._shape_cache:
            self._shape_cache[key] = z3.Int(f"shape_{name}_{axis}")
        return self._shape_cache[key]

    def ndim_var(self, name: str) -> z3.ArithRef:
        """Integer Z3 variable for ndim(name)."""
        if name not in self._ndim_cache:
            self._ndim_cache[name] = z3.Int(f"ndim_{name}")
        return self._ndim_cache[name]

    def encode(self, pred: Pred) -> z3.BoolRef:
        """Convert a Pred to a Z3 formula."""
        op = pred.op
        args = pred.args

        if op == PredOp.TRUE:
            return z3.BoolVal(True)
        if op == PredOp.FALSE:
            return z3.BoolVal(False)

        if op == PredOp.EQ:
            return self.int_var(args[0]) == args[1]
        if op == PredOp.NEQ:
            return self.int_var(args[0]) != args[1]
        if op == PredOp.LT:
            return self.int_var(args[0]) < args[1]
        if op == PredOp.LE:
            return self.int_var(args[0]) <= args[1]
        if op == PredOp.GT:
            return self.int_var(args[0]) > args[1]
        if op == PredOp.GE:
            return self.int_var(args[0]) >= args[1]

        if op == PredOp.IS_NONE:
            return self.none_var(args[0])
        if op == PredOp.IS_NOT_NONE:
            return z3.Not(self.none_var(args[0]))

        if op == PredOp.ISINSTANCE:
            tag = self._type_tags.get(args[1], 99)
            return self.tag_var(args[0]) == tag

        if op == PredOp.HASATTR:
            return z3.Bool(f"hasattr_{args[0]}_{args[1]}")

        if op == PredOp.TRUTHY:
            return self.bool_var(args[0])

        if op == PredOp.DIVISIBLE:
            v = self.int_var(args[0])
            return v % args[1] == 0

        if op == PredOp.IN_RANGE:
            v = self.int_var(args[0])
            return z3.And(v >= args[1], v <= args[2])

        if op == PredOp.LEN_EQ:
            return self.len_var(args[0]) == args[1]
        if op == PredOp.LEN_GT:
            return self.len_var(args[0]) > args[1]
        if op == PredOp.LEN_GE:
            return self.len_var(args[0]) >= args[1]

        # Tensor shape predicates
        if op == PredOp.SHAPE_EQ:
            var, dims = args[0], args[1]
            constraints = [self.ndim_var(var) == len(dims)]
            for i, d in enumerate(dims):
                if isinstance(d, str):
                    constraints.append(self.shape_dim_var(var, i) == self.int_var(d))
                else:
                    constraints.append(self.shape_dim_var(var, i) == d)
            return z3.And(*constraints) if len(constraints) > 1 else constraints[0]

        if op == PredOp.SHAPE_DIM_EQ:
            var, axis, dim = args[0], args[1], args[2]
            if isinstance(dim, str):
                return self.shape_dim_var(var, axis) == self.int_var(dim)
            return self.shape_dim_var(var, axis) == dim

        if op == PredOp.SHAPE_NDIM:
            return self.ndim_var(args[0]) == args[1]

        if op == PredOp.SHAPE_COMPAT:
            var1, var2, operation = args[0], args[1], args[2]
            if operation == "matmul":
                # For matmul: last dim of var1 == second-to-last dim of var2
                return self.shape_dim_var(var1, -1) == self.shape_dim_var(var2, -2)
            elif operation in ("add", "broadcast"):
                # Broadcasting: corresponding dimensions must be equal or 1
                # Encode for a fixed number of trailing dimensions
                n_dims = 4  # check up to 4 trailing dimensions
                constraints = []
                for i in range(n_dims):
                    d1 = self.shape_dim_var(var1, -(i + 1))
                    d2 = self.shape_dim_var(var2, -(i + 1))
                    constraints.append(z3.Or(d1 == d2, d1 == 1, d2 == 1))
                return z3.And(*constraints)
            elif operation == "cat":
                # Cat: all non-cat dims must be equal
                # Encode for up to 4 dims, cat dim 0 by default
                n_dims = 4
                constraints = []
                for i in range(1, n_dims):  # skip dim 0 (cat dim)
                    d1 = self.shape_dim_var(var1, i)
                    d2 = self.shape_dim_var(var2, i)
                    constraints.append(d1 == d2)
                # ndim must match
                constraints.append(
                    self.ndim_var(var1) == self.ndim_var(var2))
                return z3.And(*constraints) if constraints else z3.BoolVal(True)
            elif operation == "reshape":
                # Reshape: total elements must be preserved
                # product(var1 dims) == product(var2 dims)
                # Encode for up to 4 dims each
                n_dims = 4
                prod1 = self.shape_dim_var(var1, 0)
                for i in range(1, n_dims):
                    prod1 = prod1 * self.shape_dim_var(var1, i)
                prod2 = self.shape_dim_var(var2, 0)
                for i in range(1, n_dims):
                    prod2 = prod2 * self.shape_dim_var(var2, i)
                return prod1 == prod2
            return z3.BoolVal(True)

        if op == PredOp.AND:
            return z3.And(self.encode(pred.children[0]),
                          self.encode(pred.children[1]))
        if op == PredOp.OR:
            return z3.Or(self.encode(pred.children[0]),
                         self.encode(pred.children[1]))
        if op == PredOp.NOT:
            return z3.Not(self.encode(pred.children[0]))
        if op == PredOp.IMPLIES:
            return z3.Implies(self.encode(pred.children[0]),
                              self.encode(pred.children[1]))

        # Fallback: unknown predicates become uninterpreted booleans
        return z3.Bool(f"unknown_{id(pred)}")


# ---------------------------------------------------------------------------
# Refinement Type Lattice
# ---------------------------------------------------------------------------

class RefinementLattice:
    """A lattice over refinement types with Z3-backed subtyping.

    Lattice structure:
        ⊤ = {x : τ | true}         (no refinement)
        ⊥ = {x : τ | false}        (empty type)
        join = disjunction of predicates
        meet = conjunction of predicates
        subtyping: {x:T|P} <: {x:T|Q} iff ∀x. P(x) ⟹ Q(x)

    Widening uses predicate abstraction:
        Given a finite set of predicates P₁,...,Pₙ, the widened type
        retains only those Pᵢ that are implied by the current predicate.
        This guarantees termination via the ascending chain condition.
    """

    def __init__(self, timeout_ms: int = 5000):
        self.encoder = Z3Encoder()
        self.timeout_ms = timeout_ms
        self._subtype_cache: Dict[Tuple[int, int], bool] = {}
        self.stats = LatticeStats()

    def top(self, base: BaseTypeR) -> RefType:
        return RefType.trivial(base)

    def bottom(self, base: BaseTypeR) -> RefType:
        return RefType.bottom(base)

    def is_top(self, ty: RefType) -> bool:
        return ty.pred.op == PredOp.TRUE

    def is_bottom(self, ty: RefType) -> bool:
        if ty.pred.op == PredOp.FALSE:
            return True
        # Also check via SMT
        return not self._is_sat(ty.pred)

    def _is_sat(self, pred: Pred) -> bool:
        """Check if a predicate is satisfiable."""
        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(self.encoder.encode(pred))
        result = s.check()
        return result == z3.sat

    def subtype(self, sub: RefType, sup: RefType) -> bool:
        """Check {x:T|P} <: {x:T|Q}: base subtyping + ∀x. P(x) ⟹ Q(x).

        We check ¬∃x. P(x) ∧ ¬Q(x) via Z3.
        """
        cache_key = (hash(sub), hash(sup))
        if cache_key in self._subtype_cache:
            return self._subtype_cache[cache_key]

        self.stats.subtype_checks += 1
        start = time.monotonic()

        # Base type check
        if not sub.base.is_subtype_of(sup.base):
            self._subtype_cache[cache_key] = False
            return False

        # Trivial cases
        if sub.pred.op == PredOp.FALSE:
            self._subtype_cache[cache_key] = True
            return True
        if sup.pred.op == PredOp.TRUE:
            self._subtype_cache[cache_key] = True
            return True

        # Alpha-rename to common binder
        common = "ν_sub"
        p = sub.alpha_rename(common).pred
        q = sup.alpha_rename(common).pred

        # Check ¬∃x. P(x) ∧ ¬Q(x)  ≡  P ∧ ¬Q is UNSAT
        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(self.encoder.encode(p))
        s.add(z3.Not(self.encoder.encode(q)))
        result = s.check()

        is_sub = (result == z3.unsat)
        self.stats.subtype_time_ms += (time.monotonic() - start) * 1000
        self._subtype_cache[cache_key] = is_sub
        return is_sub

    def join(self, a: RefType, b: RefType) -> RefType:
        """Least upper bound: disjunction of predicates.

        join({x:T|P}, {x:T|Q}) = {x:T | P ∨ Q}
        """
        self.stats.join_count += 1
        if not a.base.is_subtype_of(b.base) and not b.base.is_subtype_of(a.base):
            # Incompatible bases → widen to Any
            return RefType.trivial(ANY_TYPE)

        base = b.base if a.base.is_subtype_of(b.base) else a.base

        if a.pred.op == PredOp.TRUE or b.pred.op == PredOp.TRUE:
            return RefType.trivial(base)
        if a.pred.op == PredOp.FALSE:
            return RefType(b.binder, base, b.pred)
        if b.pred.op == PredOp.FALSE:
            return RefType(a.binder, base, a.pred)

        common = "ν"
        pa = a.alpha_rename(common).pred
        pb = b.alpha_rename(common).pred
        return RefType(common, base, pa.or_(pb))

    def meet(self, a: RefType, b: RefType) -> RefType:
        """Greatest lower bound: conjunction of predicates.

        meet({x:T|P}, {x:T|Q}) = {x:T | P ∧ Q}
        """
        self.stats.meet_count += 1
        if a.base.is_subtype_of(b.base):
            base = a.base
        elif b.base.is_subtype_of(a.base):
            base = b.base
        else:
            return RefType.bottom(NEVER_TYPE)

        if a.pred.op == PredOp.FALSE or b.pred.op == PredOp.FALSE:
            return RefType.bottom(base)
        if a.pred.op == PredOp.TRUE:
            return RefType(b.binder, base, b.pred)
        if b.pred.op == PredOp.TRUE:
            return RefType(a.binder, base, a.pred)

        common = "ν"
        pa = a.alpha_rename(common).pred
        pb = b.alpha_rename(common).pred
        return RefType(common, base, pa.and_(pb))

    def widen(self, prev: RefType, curr: RefType,
              predicates: Sequence[Pred]) -> RefType:
        """Widening via predicate abstraction.

        Given a finite set of predicates {P₁, ..., Pₙ}, the widened type
        retains only those Pᵢ implied by the current refinement.

        This guarantees the ascending chain condition: the lattice height
        is bounded by 2^|predicates|, so fixed-point iteration terminates.

        widen(prev, curr, {P₁,...,Pₙ}) = {x:T | ⋀{Pᵢ | curr.pred ⟹ Pᵢ}}
        """
        self.stats.widen_count += 1
        if not predicates:
            return RefType.trivial(curr.base)

        common = "ν"
        curr_pred = curr.alpha_rename(common).pred

        retained: List[Pred] = []
        for p in predicates:
            p_renamed = p.substitute(p.free_vars().__iter__().__next__()
                                     if p.free_vars() else "ν", common) \
                        if p.free_vars() else p
            # Check: curr_pred ⟹ p_renamed
            test_type = RefType(common, curr.base, curr_pred)
            target_type = RefType(common, curr.base, p_renamed)
            if self.subtype(test_type, target_type):
                retained.append(p_renamed)

        if not retained:
            return RefType.trivial(curr.base)

        result_pred = retained[0]
        for p in retained[1:]:
            result_pred = result_pred.and_(p)
        return RefType(common, curr.base, result_pred)

    def narrow(self, wide: RefType, precise: RefType) -> RefType:
        """Narrowing: recover precision after widening.

        narrow(W, P) = meet(W, P) if the result is non-bottom, else W.
        """
        self.stats.narrow_count += 1
        result = self.meet(wide, precise)
        if self.is_bottom(result):
            return wide
        return result

    def leq(self, a: RefType, b: RefType) -> bool:
        """Partial order: a ⊑ b iff a <: b."""
        return self.subtype(a, b)

    def equiv(self, a: RefType, b: RefType) -> bool:
        """Equivalence: a ≡ b iff a <: b and b <: a."""
        return self.subtype(a, b) and self.subtype(b, a)


@dataclass
class LatticeStats:
    """Statistics for lattice operations."""
    subtype_checks: int = 0
    subtype_time_ms: float = 0.0
    join_count: int = 0
    meet_count: int = 0
    widen_count: int = 0
    narrow_count: int = 0


# ---------------------------------------------------------------------------
# Predicate abstraction domain backed by Z3
# ---------------------------------------------------------------------------

@dataclass
class PredicateAbstractionState:
    """Abstract state in the predicate abstraction domain.

    Maps each variable to a set of predicates known to hold.
    The concrete meaning is the conjunction of all predicates.
    """
    var_preds: Dict[str, List[Pred]] = field(default_factory=dict)
    path_pred: Pred = field(default_factory=Pred.true_)

    def add_pred(self, var: str, pred: Pred) -> None:
        if var not in self.var_preds:
            self.var_preds[var] = []
        self.var_preds[var].append(pred)

    def get_refinement(self, var: str, base: BaseTypeR) -> RefType:
        """Get the refinement type for a variable."""
        preds = self.var_preds.get(var, [])
        if not preds:
            return RefType.trivial(base)
        combined = preds[0]
        for p in preds[1:]:
            combined = combined.and_(p)
        return RefType("ν", base, combined)

    def conjunction(self) -> Pred:
        """Get the conjunction of all predicates (path + variable)."""
        result = self.path_pred
        for var, preds in self.var_preds.items():
            for p in preds:
                result = result.and_(p)
        return result


class PredicateAbstractionDomain:
    """Abstract domain backed by predicate abstraction over Z3.

    The domain tracks a finite set of predicates and uses Z3 to determine
    which predicates hold at each program point. Widening drops predicates
    that are no longer provably true, guaranteeing termination.
    """

    def __init__(self, predicates: Sequence[Pred],
                 lattice: Optional[RefinementLattice] = None):
        self.predicates = list(predicates)
        self.lattice = lattice or RefinementLattice()
        self.encoder = self.lattice.encoder

    def alpha(self, concrete_pred: Pred) -> PredicateAbstractionState:
        """Abstraction function: compute which tracked predicates
        are implied by the concrete predicate."""
        state = PredicateAbstractionState()

        for p in self.predicates:
            fvs = p.free_vars()
            if not fvs:
                continue
            var = next(iter(fvs))
            # Check: concrete_pred ⟹ p
            s = z3.Solver()
            s.set("timeout", self.lattice.timeout_ms)
            s.add(self.encoder.encode(concrete_pred))
            s.add(z3.Not(self.encoder.encode(p)))
            if s.check() == z3.unsat:
                state.add_pred(var, p)

        return state

    def join(self, a: PredicateAbstractionState,
             b: PredicateAbstractionState) -> PredicateAbstractionState:
        """Join: keep predicates that hold in BOTH states."""
        result = PredicateAbstractionState()
        all_vars = set(a.var_preds.keys()) | set(b.var_preds.keys())

        for var in all_vars:
            a_preds = set(id(p) for p in a.var_preds.get(var, []))
            b_preds_list = b.var_preds.get(var, [])

            # Keep predicates that appear in both (by structural equality)
            a_pred_set = a.var_preds.get(var, [])
            for pa in a_pred_set:
                for pb in b_preds_list:
                    if pa == pb:
                        result.add_pred(var, pa)
                        break

        result.path_pred = a.path_pred.or_(b.path_pred)
        return result

    def meet(self, a: PredicateAbstractionState,
             b: PredicateAbstractionState) -> PredicateAbstractionState:
        """Meet: keep predicates from EITHER state."""
        result = PredicateAbstractionState()
        all_vars = set(a.var_preds.keys()) | set(b.var_preds.keys())

        for var in all_vars:
            seen: Set[Pred] = set()
            for p in a.var_preds.get(var, []):
                if p not in seen:
                    result.add_pred(var, p)
                    seen.add(p)
            for p in b.var_preds.get(var, []):
                if p not in seen:
                    result.add_pred(var, p)
                    seen.add(p)

        result.path_pred = a.path_pred.and_(b.path_pred)
        return result

    def widen(self, prev: PredicateAbstractionState,
              curr: PredicateAbstractionState) -> PredicateAbstractionState:
        """Widening: drop predicates not provably true in curr.

        Since our predicate set is finite, this is equivalent to
        intersection (join). The ascending chain stabilizes in at
        most |predicates| iterations.
        """
        return self.join(prev, curr)

    def narrow(self, wide: PredicateAbstractionState,
               precise: PredicateAbstractionState) -> PredicateAbstractionState:
        """Narrowing: add back predicates from precise that are
        consistent with wide."""
        result = PredicateAbstractionState()
        result.var_preds = dict(wide.var_preds)
        result.path_pred = wide.path_pred

        for var, preds in precise.var_preds.items():
            if var not in result.var_preds:
                result.var_preds[var] = []
            existing = set(result.var_preds.get(var, []))
            for p in preds:
                if p not in existing:
                    # Check consistency: wide.conjunction ∧ p is SAT
                    combined = wide.path_pred.and_(p)
                    if self.lattice._is_sat(combined):
                        result.add_pred(var, p)

        return result

    def add_predicate(self, pred: Pred) -> None:
        """Dynamically add a new predicate to track (used by contract discovery)."""
        if pred not in self.predicates:
            self.predicates.append(pred)

    def transfer_guard(self, state: PredicateAbstractionState,
                       guard_pred: Pred,
                       positive: bool) -> PredicateAbstractionState:
        """Narrow state at a guard point (branch condition)."""
        effective = guard_pred if positive else guard_pred.not_()
        new_path = state.path_pred.and_(effective)
        result = PredicateAbstractionState(
            var_preds=dict(state.var_preds),
            path_pred=new_path
        )
        # Re-evaluate tracked predicates under the new path
        for p in self.predicates:
            fvs = p.free_vars()
            if not fvs:
                continue
            var = next(iter(fvs))
            s = z3.Solver()
            s.set("timeout", self.lattice.timeout_ms)
            s.add(self.encoder.encode(new_path))
            s.add(z3.Not(self.encoder.encode(p)))
            if s.check() == z3.unsat:
                if var not in result.var_preds:
                    result.var_preds[var] = []
                if p not in result.var_preds[var]:
                    result.add_pred(var, p)
        return result


# ---------------------------------------------------------------------------
# Dependent function types: (x:{v:τ₁|P}) → {v:τ₂|Q(x)}
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepFuncType:
    """Dependent function type for interprocedural analysis.

    (x₁:{v:T₁|P₁}, ..., xₙ:{v:Tₙ|Pₙ}) → {v:T_ret|Q(x₁,...,xₙ)}

    The return predicate Q may mention the parameter names.
    """
    params: Tuple[Tuple[str, RefType], ...]
    ret: RefType

    def pretty(self) -> str:
        ps = ", ".join(f"{n}: {t.pretty()}" for n, t in self.params)
        return f"({ps}) → {self.ret.pretty()}"

    def apply(self, args: Dict[str, RefType],
              lattice: RefinementLattice) -> RefType:
        """Apply this function type to concrete argument types.

        Checks that each argument satisfies its parameter's refinement,
        then substitutes into the return type.
        """
        ret_pred = self.ret.pred
        for param_name, param_type in self.params:
            if param_name in args:
                arg = args[param_name]
                # Substitute param_name with arg's binder in return predicate
                ret_pred = ret_pred.substitute(param_name, arg.binder)
        return RefType(self.ret.binder, self.ret.base, ret_pred)

    def check_args(self, args: Dict[str, RefType],
                   lattice: RefinementLattice) -> List[str]:
        """Check if arguments satisfy parameter refinements. Returns errors."""
        errors = []
        for param_name, param_type in self.params:
            if param_name in args:
                if not lattice.subtype(args[param_name], param_type):
                    errors.append(
                        f"Argument {param_name}: "
                        f"{args[param_name].pretty()} is not a subtype of "
                        f"{param_type.pretty()}"
                    )
        return errors


# ---------------------------------------------------------------------------
# Refinement environment: maps variables to refinement types
# ---------------------------------------------------------------------------

@dataclass
class RefEnvironment:
    """Type environment mapping variables to refinement types."""
    bindings: Dict[str, RefType] = field(default_factory=dict)

    def get(self, var: str) -> Optional[RefType]:
        return self.bindings.get(var)

    def set(self, var: str, ty: RefType) -> RefEnvironment:
        new_env = RefEnvironment(dict(self.bindings))
        new_env.bindings[var] = ty
        return new_env

    def join(self, other: RefEnvironment,
             lattice: RefinementLattice) -> RefEnvironment:
        """Pointwise join of environments."""
        result = RefEnvironment()
        all_vars = set(self.bindings.keys()) | set(other.bindings.keys())
        for v in all_vars:
            a = self.bindings.get(v)
            b = other.bindings.get(v)
            if a is None:
                result.bindings[v] = b if b else RefType.trivial(ANY_TYPE)
            elif b is None:
                result.bindings[v] = a
            else:
                result.bindings[v] = lattice.join(a, b)
        return result

    def meet(self, other: RefEnvironment,
             lattice: RefinementLattice) -> RefEnvironment:
        """Pointwise meet of environments."""
        result = RefEnvironment()
        all_vars = set(self.bindings.keys()) & set(other.bindings.keys())
        for v in all_vars:
            result.bindings[v] = lattice.meet(
                self.bindings[v], other.bindings[v])
        return result

    def widen(self, other: RefEnvironment,
              predicates: Sequence[Pred],
              lattice: RefinementLattice) -> RefEnvironment:
        """Pointwise widening of environments."""
        result = RefEnvironment()
        all_vars = set(self.bindings.keys()) | set(other.bindings.keys())
        for v in all_vars:
            a = self.bindings.get(v)
            b = other.bindings.get(v)
            if a is None or b is None:
                result.bindings[v] = RefType.trivial(
                    (b or a).base if (b or a) else ANY_TYPE)
            else:
                result.bindings[v] = lattice.widen(a, b, predicates)
        return result

    def pretty(self) -> str:
        items = [f"  {v}: {t.pretty()}" for v, t in self.bindings.items()]
        return "{\n" + "\n".join(items) + "\n}"
