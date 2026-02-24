"""
Liquid type system implementation.

Provides AST nodes for liquid predicates, liquid types with refinement
predicates, dependent function types, subtyping constraint generation,
bidirectional type inference, and fixpoint solving for recursive types.
"""

from __future__ import annotations

import copy
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
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Local type stubs – no cross-module imports
# ---------------------------------------------------------------------------

# Minimal stand-ins for an external program IR that the constraint generator
# walks.  Real consumers would supply their own IR nodes.

class _IRNodeKind(Enum):
    VAR = auto()
    CONST = auto()
    APP = auto()
    LET = auto()
    IF = auto()
    FUN = auto()
    RETURN = auto()
    ASSIGN = auto()
    SEQ = auto()
    ASSERT = auto()


@dataclass
class IRNode:
    """Minimal program IR node used by the constraint generator."""
    kind: _IRNodeKind
    name: Optional[str] = None
    value: Any = None
    children: List["IRNode"] = field(default_factory=list)
    annotation: Optional[Any] = None


# ===================================================================
# 1. AST for liquid predicates (refinement expressions)
# ===================================================================

class LiquidExpr:
    """Base class for refinement-predicate AST nodes."""

    def free_vars(self) -> Set[str]:
        raise NotImplementedError

    def substitute(self, mapping: Dict[str, "LiquidExpr"]) -> "LiquidExpr":
        raise NotImplementedError

    def evaluate(self, env: Dict[str, Any]) -> Any:
        """Evaluate under a concrete environment (for testing / CE extraction)."""
        raise NotImplementedError

    def negate(self) -> "LiquidExpr":
        return LiquidApp("not", [self])

    def __and__(self, other: "LiquidExpr") -> "LiquidExpr":
        return LiquidApp("and", [self, other])

    def __or__(self, other: "LiquidExpr") -> "LiquidExpr":
        return LiquidApp("or", [self, other])


@dataclass(frozen=True)
class LiquidVar(LiquidExpr):
    """A variable reference inside a refinement predicate."""
    name: str

    def free_vars(self) -> Set[str]:
        return {self.name}

    def substitute(self, mapping: Dict[str, LiquidExpr]) -> LiquidExpr:
        return mapping.get(self.name, self)

    def evaluate(self, env: Dict[str, Any]) -> Any:
        if self.name not in env:
            raise ValueError(f"Unbound variable {self.name}")
        return env[self.name]

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class LiquidConst(LiquidExpr):
    """A literal constant (int, bool, string, None)."""
    value: Any

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, mapping: Dict[str, LiquidExpr]) -> LiquidExpr:
        return self

    def evaluate(self, env: Dict[str, Any]) -> Any:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)


# Built-in operator semantics used by LiquidApp.evaluate
_BUILTIN_OPS: Dict[str, Callable[..., Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b if b != 0 else None,
    "%": lambda a, b: a % b if b != 0 else None,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "not": lambda a: not a,
    "implies": lambda a, b: (not a) or b,
    "len": lambda a: len(a),
    "neg": lambda a: -a,
}


@dataclass(frozen=True)
class LiquidApp(LiquidExpr):
    """Application of a function / operator to arguments."""
    func: str
    args: Tuple[LiquidExpr, ...]

    def __init__(self, func: str, args: Union[List[LiquidExpr], Tuple[LiquidExpr, ...]]):
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "args", tuple(args))

    def free_vars(self) -> Set[str]:
        result: Set[str] = set()
        for a in self.args:
            result |= a.free_vars()
        return result

    def substitute(self, mapping: Dict[str, LiquidExpr]) -> LiquidExpr:
        new_args = [a.substitute(mapping) for a in self.args]
        return LiquidApp(self.func, new_args)

    def evaluate(self, env: Dict[str, Any]) -> Any:
        vals = [a.evaluate(env) for a in self.args]
        op = _BUILTIN_OPS.get(self.func)
        if op is None:
            raise ValueError(f"Unknown operator {self.func}")
        return op(*vals)

    def __repr__(self) -> str:
        if len(self.args) == 1:
            return f"({self.func} {self.args[0]})"
        if len(self.args) == 2:
            return f"({self.args[0]} {self.func} {self.args[1]})"
        arg_str = ", ".join(repr(a) for a in self.args)
        return f"{self.func}({arg_str})"


# Convenience constructors
TRUE = LiquidConst(True)
FALSE = LiquidConst(False)


def liquid_and(exprs: List[LiquidExpr]) -> LiquidExpr:
    """Conjunction of a list of expressions."""
    filtered = [e for e in exprs if e != TRUE]
    if not filtered:
        return TRUE
    if FALSE in filtered:
        return FALSE
    result = filtered[0]
    for e in filtered[1:]:
        result = LiquidApp("and", [result, e])
    return result


def liquid_or(exprs: List[LiquidExpr]) -> LiquidExpr:
    """Disjunction of a list of expressions."""
    filtered = [e for e in exprs if e != FALSE]
    if not filtered:
        return FALSE
    if TRUE in filtered:
        return TRUE
    result = filtered[0]
    for e in filtered[1:]:
        result = LiquidApp("or", [result, e])
    return result


def liquid_implies(lhs: LiquidExpr, rhs: LiquidExpr) -> LiquidExpr:
    return LiquidApp("implies", [lhs, rhs])


# ===================================================================
# 2. Base types
# ===================================================================

class BaseType(Enum):
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    NONE_TYPE = "none"
    ANY = "any"


# ===================================================================
# 3. Liquid types
# ===================================================================

@dataclass
class LiquidType:
    """
    A liquid type  {x : T | e}  where *T* is a base type and *e* is a
    boolean refinement expression that may mention *x*.
    """
    var: str
    base: BaseType
    predicate: LiquidExpr

    @staticmethod
    def trivial(base: BaseType, var: str = "v") -> "LiquidType":
        """Create an unrefined type ``{v : T | true}``."""
        return LiquidType(var=var, base=base, predicate=TRUE)

    def substitute_predicate(self, mapping: Dict[str, LiquidExpr]) -> "LiquidType":
        """Return a copy with *mapping* applied to the predicate."""
        return LiquidType(
            var=self.var,
            base=self.base,
            predicate=self.predicate.substitute(mapping),
        )

    def rename_var(self, new_var: str) -> "LiquidType":
        """Alpha-rename the bound variable."""
        if new_var == self.var:
            return self
        new_pred = self.predicate.substitute({self.var: LiquidVar(new_var)})
        return LiquidType(var=new_var, base=self.base, predicate=new_pred)

    def strengthen(self, extra: LiquidExpr) -> "LiquidType":
        """Return a type with the predicate conjoined with *extra*."""
        return LiquidType(
            var=self.var,
            base=self.base,
            predicate=liquid_and([self.predicate, extra]),
        )

    def free_vars(self) -> Set[str]:
        fv = self.predicate.free_vars()
        fv.discard(self.var)
        return fv

    def __repr__(self) -> str:
        return f"{{{self.var}:{self.base.value} | {self.predicate}}}"


@dataclass
class LiquidFunctionType:
    """
    Dependent function type  (x₁:T₁, x₂:T₂, …) → T_ret
    where each Tᵢ and T_ret may mention earlier parameters.
    """
    params: List[Tuple[str, LiquidType]]
    ret: LiquidType

    def arity(self) -> int:
        return len(self.params)

    def substitute(self, mapping: Dict[str, LiquidExpr]) -> "LiquidFunctionType":
        new_params: List[Tuple[str, LiquidType]] = []
        cur_map = dict(mapping)
        for name, ty in self.params:
            new_ty = ty.substitute_predicate(cur_map)
            new_params.append((name, new_ty))
            # shadowing
            cur_map.pop(name, None)
        new_ret = self.ret.substitute_predicate(cur_map)
        return LiquidFunctionType(params=new_params, ret=new_ret)

    def free_vars(self) -> Set[str]:
        fv: Set[str] = set()
        bound: Set[str] = set()
        for name, ty in self.params:
            fv |= ty.free_vars() - bound
            bound.add(name)
        fv |= self.ret.free_vars() - bound
        return fv

    def __repr__(self) -> str:
        ps = ", ".join(f"{n}:{t}" for n, t in self.params)
        return f"({ps}) -> {self.ret}"


AnyLiquidType = Union[LiquidType, LiquidFunctionType]


# ===================================================================
# 4. Subtyping constraints and Horn clauses
# ===================================================================

@dataclass
class SubtypingConstraint:
    """
    Represents the obligation  Γ ⊢ {x:T | p} <: {x:T | q}.

    *environment* maps variable names to their liquid types.
    """
    environment: Dict[str, LiquidType]
    lhs: LiquidType
    rhs: LiquidType
    source_location: Optional[str] = None

    def to_horn_clause(self) -> "HornClause":
        """Lower this subtyping check into a Horn clause.

        The clause says: if all environment predicates hold *and* the LHS
        predicate holds, then the RHS predicate must hold (with the bound
        variable unified).
        """
        premises: List[LiquidExpr] = []
        for var_name, ltype in self.environment.items():
            prem = ltype.predicate.substitute({ltype.var: LiquidVar(var_name)})
            premises.append(prem)
        # rename lhs/rhs bound variable to a fresh name
        fresh = self.lhs.var
        lhs_pred = self.lhs.predicate  # already uses fresh
        rhs_pred = self.rhs.predicate.substitute({self.rhs.var: LiquidVar(fresh)})
        premises.append(lhs_pred)
        return HornClause(body=premises, head=rhs_pred)

    def __repr__(self) -> str:
        env_str = ", ".join(f"{k}:{v}" for k, v in self.environment.items())
        return f"[{env_str}] ⊢ {self.lhs} <: {self.rhs}"


@dataclass
class HornClause:
    """
    Horn clause encoding:  p₁ ∧ p₂ ∧ … → q

    Used as the fundamental unit consumed by the fixpoint solver.
    """
    body: List[LiquidExpr]
    head: LiquidExpr

    def body_conjunction(self) -> LiquidExpr:
        return liquid_and(self.body)

    def free_vars(self) -> Set[str]:
        fv: Set[str] = set()
        for b in self.body:
            fv |= b.free_vars()
        fv |= self.head.free_vars()
        return fv

    def substitute(self, mapping: Dict[str, LiquidExpr]) -> "HornClause":
        new_body = [b.substitute(mapping) for b in self.body]
        new_head = self.head.substitute(mapping)
        return HornClause(body=new_body, head=new_head)

    def is_trivially_valid(self) -> bool:
        """Quick syntactic check: head ∈ body."""
        for b in self.body:
            if b == self.head:
                return True
        if self.head == TRUE:
            return True
        return False

    def __repr__(self) -> str:
        body_str = " ∧ ".join(repr(b) for b in self.body) if self.body else "⊤"
        return f"{body_str}  →  {self.head}"


# ===================================================================
# 5. Typing environment
# ===================================================================

class TypingEnvironment:
    """Immutable-style environment mapping names to liquid types."""

    def __init__(self, bindings: Optional[Dict[str, AnyLiquidType]] = None):
        self._bindings: Dict[str, AnyLiquidType] = dict(bindings) if bindings else {}

    def lookup(self, name: str) -> Optional[AnyLiquidType]:
        return self._bindings.get(name)

    def extend(self, name: str, ty: AnyLiquidType) -> "TypingEnvironment":
        new_bindings = dict(self._bindings)
        new_bindings[name] = ty
        return TypingEnvironment(new_bindings)

    def remove(self, name: str) -> "TypingEnvironment":
        new_bindings = dict(self._bindings)
        new_bindings.pop(name, None)
        return TypingEnvironment(new_bindings)

    def domain(self) -> Set[str]:
        return set(self._bindings.keys())

    def base_types_only(self) -> Dict[str, LiquidType]:
        """Return only LiquidType entries (not function types)."""
        return {k: v for k, v in self._bindings.items() if isinstance(v, LiquidType)}

    def items(self):
        return self._bindings.items()

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}: {v}" for k, v in self._bindings.items())
        return f"Γ[{inner}]"


# ===================================================================
# 6. Constraint generator
# ===================================================================

class ConstraintGenerator:
    """
    Walks program IR and emits :class:`SubtypingConstraint` obligations.

    The generator maintains a typing environment and a fresh-name counter.
    """

    def __init__(self) -> None:
        self._constraints: List[SubtypingConstraint] = []
        self._counter: int = 0
        self._env: TypingEnvironment = TypingEnvironment()
        self._function_sigs: Dict[str, LiquidFunctionType] = {}

    # -- helpers ----------------------------------------------------------

    def _fresh(self, prefix: str = "v") -> str:
        self._counter += 1
        return f"__{prefix}_{self._counter}"

    def _emit(self, lhs: LiquidType, rhs: LiquidType, loc: Optional[str] = None) -> None:
        sc = SubtypingConstraint(
            environment=self._env.base_types_only(),
            lhs=lhs,
            rhs=rhs,
            source_location=loc,
        )
        self._constraints.append(sc)

    def constraints(self) -> List[SubtypingConstraint]:
        return list(self._constraints)

    def horn_clauses(self) -> List[HornClause]:
        return [c.to_horn_clause() for c in self._constraints]

    # -- public entry point -----------------------------------------------

    def generate(self, node: IRNode, env: Optional[TypingEnvironment] = None) -> AnyLiquidType:
        """Generate constraints for *node* under *env* and return inferred type."""
        if env is not None:
            self._env = env
        return self._gen(node)

    # -- recursive dispatch -----------------------------------------------

    def _gen(self, node: IRNode) -> AnyLiquidType:
        dispatch = {
            _IRNodeKind.VAR: self._gen_var,
            _IRNodeKind.CONST: self._gen_const,
            _IRNodeKind.APP: self._gen_app,
            _IRNodeKind.LET: self._gen_let,
            _IRNodeKind.IF: self._gen_if,
            _IRNodeKind.FUN: self._gen_fun,
            _IRNodeKind.RETURN: self._gen_return,
            _IRNodeKind.ASSIGN: self._gen_assign,
            _IRNodeKind.SEQ: self._gen_seq,
            _IRNodeKind.ASSERT: self._gen_assert,
        }
        handler = dispatch.get(node.kind)
        if handler is None:
            raise ValueError(f"Unknown IR node kind: {node.kind}")
        return handler(node)

    # -- node handlers ----------------------------------------------------

    def _gen_var(self, node: IRNode) -> AnyLiquidType:
        """Variable reference: look up in environment."""
        assert node.name is not None
        ty = self._env.lookup(node.name)
        if ty is None:
            return LiquidType.trivial(BaseType.ANY, var=self._fresh())
        return ty

    def _gen_const(self, node: IRNode) -> AnyLiquidType:
        """Constant: infer a singleton liquid type."""
        v = self._fresh()
        if isinstance(node.value, bool):
            base = BaseType.BOOL
            pred = LiquidApp("==", [LiquidVar(v), LiquidConst(node.value)])
        elif isinstance(node.value, int):
            base = BaseType.INT
            pred = LiquidApp("==", [LiquidVar(v), LiquidConst(node.value)])
        elif isinstance(node.value, str):
            base = BaseType.STRING
            pred = LiquidApp("==", [LiquidVar(v), LiquidConst(node.value)])
        elif node.value is None:
            base = BaseType.NONE_TYPE
            pred = TRUE
        else:
            base = BaseType.ANY
            pred = TRUE
        return LiquidType(var=v, base=base, predicate=pred)

    def _gen_app(self, node: IRNode) -> AnyLiquidType:
        """Function / operator application."""
        assert len(node.children) >= 1
        func_node = node.children[0]
        arg_nodes = node.children[1:]

        func_ty = self._gen(func_node)
        if not isinstance(func_ty, LiquidFunctionType):
            # treat as untyped application
            for arg in arg_nodes:
                self._gen(arg)
            return LiquidType.trivial(BaseType.ANY, var=self._fresh())

        # Check arity
        if len(arg_nodes) != func_ty.arity():
            for arg in arg_nodes:
                self._gen(arg)
            return LiquidType.trivial(BaseType.ANY, var=self._fresh())

        # Generate subtyping for each argument
        mapping: Dict[str, LiquidExpr] = {}
        for (param_name, param_ty), arg_node in zip(func_ty.params, arg_nodes):
            arg_ty = self._gen(arg_node)
            if isinstance(arg_ty, LiquidType):
                expected = param_ty.substitute_predicate(mapping)
                self._emit(arg_ty, expected, loc=node.name)
                # bind actual arg to param name for dependent return type
                if arg_node.kind == _IRNodeKind.VAR and arg_node.name:
                    mapping[param_name] = LiquidVar(arg_node.name)
                else:
                    mapping[param_name] = LiquidVar(arg_ty.var)

        ret = func_ty.ret.substitute_predicate(mapping)
        return ret

    def _gen_let(self, node: IRNode) -> AnyLiquidType:
        """let x = e1 in e2."""
        assert node.name is not None
        assert len(node.children) == 2
        rhs_ty = self._gen(node.children[0])
        old_env = self._env
        self._env = self._env.extend(node.name, rhs_ty)
        body_ty = self._gen(node.children[1])
        self._env = old_env
        return body_ty

    def _gen_if(self, node: IRNode) -> AnyLiquidType:
        """if cond then e1 else e2."""
        assert len(node.children) in (2, 3)
        cond_ty = self._gen(node.children[0])

        # strengthen environment for each branch
        guard_pred: LiquidExpr = TRUE
        if isinstance(cond_ty, LiquidType):
            guard_pred = cond_ty.predicate.substitute(
                {cond_ty.var: LiquidVar(cond_ty.var)}
            )

        old_env = self._env
        # then branch – assume guard true
        then_ty = self._gen(node.children[1])

        # else branch
        if len(node.children) == 3:
            self._env = old_env
            else_ty = self._gen(node.children[2])
        else:
            else_ty = LiquidType.trivial(BaseType.NONE_TYPE, var=self._fresh())

        # join: create a fresh type variable that is a supertype of both
        join_var = self._fresh("join")
        if isinstance(then_ty, LiquidType) and isinstance(else_ty, LiquidType):
            join_base = then_ty.base if then_ty.base == else_ty.base else BaseType.ANY
            join_ty = LiquidType.trivial(join_base, var=join_var)
            self._emit(then_ty, join_ty, loc="if-then")
            self._emit(else_ty, join_ty, loc="if-else")
            self._env = old_env
            return join_ty

        self._env = old_env
        return then_ty

    def _gen_fun(self, node: IRNode) -> AnyLiquidType:
        """Lambda / function definition."""
        assert node.name is not None  # function name
        params_ir = node.children[:-1]
        body_ir = node.children[-1]

        old_env = self._env
        param_list: List[Tuple[str, LiquidType]] = []
        for p in params_ir:
            pname = p.name or self._fresh("p")
            pty = LiquidType.trivial(BaseType.ANY, var=self._fresh())
            if p.annotation is not None and isinstance(p.annotation, LiquidType):
                pty = p.annotation
            param_list.append((pname, pty))
            self._env = self._env.extend(pname, pty)

        body_ty = self._gen(body_ir)
        ret_ty = body_ty if isinstance(body_ty, LiquidType) else LiquidType.trivial(BaseType.ANY, var=self._fresh())

        func_ty = LiquidFunctionType(params=param_list, ret=ret_ty)
        self._env = old_env.extend(node.name, func_ty)
        self._function_sigs[node.name] = func_ty
        return func_ty

    def _gen_return(self, node: IRNode) -> AnyLiquidType:
        """Return statement – just generate for the child."""
        assert len(node.children) == 1
        return self._gen(node.children[0])

    def _gen_assign(self, node: IRNode) -> AnyLiquidType:
        """Assignment  x = e."""
        assert node.name is not None
        assert len(node.children) == 1
        rhs_ty = self._gen(node.children[0])
        self._env = self._env.extend(node.name, rhs_ty)
        return rhs_ty

    def _gen_seq(self, node: IRNode) -> AnyLiquidType:
        """Sequence of statements – generate left to right, return last type."""
        result: AnyLiquidType = LiquidType.trivial(BaseType.NONE_TYPE, var=self._fresh())
        for child in node.children:
            result = self._gen(child)
        return result

    def _gen_assert(self, node: IRNode) -> AnyLiquidType:
        """Assert – the condition must be provably true."""
        assert len(node.children) == 1
        cond_ty = self._gen(node.children[0])
        if isinstance(cond_ty, LiquidType):
            # assert subtype of {v:bool | v == true}
            expected = LiquidType(
                var=cond_ty.var,
                base=BaseType.BOOL,
                predicate=LiquidApp("==", [LiquidVar(cond_ty.var), LiquidConst(True)]),
            )
            self._emit(cond_ty, expected, loc="assert")
        return LiquidType.trivial(BaseType.NONE_TYPE, var=self._fresh())


# ===================================================================
# 7. Bidirectional inference
# ===================================================================

class BidirectionalInference:
    """
    Bidirectional liquid type inference.

    * **check** mode pushes an expected type *down* into the term.
    * **synth** mode infers a type *up* from the term.
    * **subsumes** emits subtyping constraints between two types.
    """

    def __init__(self) -> None:
        self._cgen = ConstraintGenerator()
        self._counter: int = 0

    def _fresh(self, prefix: str = "bi") -> str:
        self._counter += 1
        return f"__{prefix}_{self._counter}"

    # ---- synthesis mode -------------------------------------------------

    def synth(self, node: IRNode, env: Optional[TypingEnvironment] = None) -> AnyLiquidType:
        """Synthesise a liquid type for *node*."""
        return self._cgen.generate(node, env=env)

    # ---- checking mode --------------------------------------------------

    def check(
        self,
        node: IRNode,
        expected: AnyLiquidType,
        env: Optional[TypingEnvironment] = None,
    ) -> List[SubtypingConstraint]:
        """
        Check that *node* has type *expected*, returning generated constraints.

        Falls back to synth + subsumes when no special checking rule applies.
        """
        inferred = self.synth(node, env=env)
        self.subsumes(inferred, expected)
        return self._cgen.constraints()

    # ---- subsumption ----------------------------------------------------

    def subsumes(
        self,
        inferred: AnyLiquidType,
        expected: AnyLiquidType,
    ) -> List[SubtypingConstraint]:
        """
        Emit constraints ensuring *inferred* <: *expected*.

        For base liquid types this is a single subtyping constraint.
        For function types we use contravariant arguments / covariant return.
        """
        if isinstance(inferred, LiquidType) and isinstance(expected, LiquidType):
            self._cgen._emit(inferred, expected, loc="subsumes")
        elif isinstance(inferred, LiquidFunctionType) and isinstance(expected, LiquidFunctionType):
            self._subsumes_func(inferred, expected)
        return self._cgen.constraints()

    def _subsumes_func(
        self,
        inferred: LiquidFunctionType,
        expected: LiquidFunctionType,
    ) -> None:
        """Function subtyping: contravariant params, covariant return."""
        if inferred.arity() != expected.arity():
            return
        mapping: Dict[str, LiquidExpr] = {}
        for (iname, ity), (ename, ety) in zip(inferred.params, expected.params):
            # contravariant: expected param <: inferred param
            self._cgen._emit(ety, ity, loc="func-param-contra")
            mapping[iname] = LiquidVar(ename)
        # covariant return
        iret = inferred.ret.substitute_predicate(mapping)
        self._cgen._emit(iret, expected.ret, loc="func-ret-co")

    # ---- accessors ------------------------------------------------------

    def constraints(self) -> List[SubtypingConstraint]:
        return self._cgen.constraints()

    def horn_clauses(self) -> List[HornClause]:
        return self._cgen.horn_clauses()


# ===================================================================
# 8. Fixpoint solver for recursive liquid types
# ===================================================================

@dataclass
class _TypeVarInfo:
    """Internal bookkeeping for a type variable in the fixpoint solver."""
    name: str
    current_type: LiquidType
    predecessors: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    dirty: bool = True


class FixpointSolver:
    """
    Iterative fixpoint computation for recursive liquid types.

    Uses a worklist algorithm over type variables.  At each step the solver
    picks a dirty type variable, recomputes its type from incoming
    constraints, and marks dependents dirty if the type changed.

    Supports widening to ensure termination for recursive types.
    """

    MAX_ITERATIONS: int = 200
    WIDENING_THRESHOLD: int = 10  # widen after this many updates to same var

    def __init__(self) -> None:
        self._vars: Dict[str, _TypeVarInfo] = {}
        self._update_counts: Dict[str, int] = {}
        self._horn_clauses: List[HornClause] = []
        self._solution: Dict[str, LiquidType] = {}

    # -- API --------------------------------------------------------------

    def add_type_variable(self, name: str, initial: LiquidType) -> None:
        """Register a type variable with its initial approximation."""
        self._vars[name] = _TypeVarInfo(name=name, current_type=initial)
        self._update_counts[name] = 0

    def add_dependency(self, src: str, dst: str) -> None:
        """Declare that *dst*'s type depends on *src*'s type."""
        if src in self._vars and dst in self._vars:
            self._vars[src].successors.add(dst)
            self._vars[dst].predecessors.add(src)

    def add_horn_clauses(self, clauses: List[HornClause]) -> None:
        self._horn_clauses.extend(clauses)

    def solve(self) -> Dict[str, LiquidType]:
        """Run the worklist algorithm to a fixpoint and return solution."""
        worklist: List[str] = [n for n, info in self._vars.items() if info.dirty]
        iterations = 0

        while worklist and iterations < self.MAX_ITERATIONS:
            iterations += 1
            name = worklist.pop(0)
            info = self._vars[name]
            if not info.dirty:
                continue
            info.dirty = False

            new_type = self._recompute(name)
            if new_type is None:
                continue

            self._update_counts[name] = self._update_counts.get(name, 0) + 1

            # apply widening if too many updates
            if self._update_counts[name] > self.WIDENING_THRESHOLD:
                new_type = self._widen(info.current_type, new_type)

            if not self._types_equal(info.current_type, new_type):
                info.current_type = new_type
                for succ in info.successors:
                    self._vars[succ].dirty = True
                    if succ not in worklist:
                        worklist.append(succ)

        self._solution = {n: info.current_type for n, info in self._vars.items()}
        return dict(self._solution)

    def solution(self) -> Dict[str, LiquidType]:
        return dict(self._solution)

    def converged(self) -> bool:
        return all(not info.dirty for info in self._vars.values())

    # -- internals --------------------------------------------------------

    def _recompute(self, name: str) -> Optional[LiquidType]:
        """Recompute the type of variable *name* from its predecessors.

        Collects all predicates implied by predecessor types and relevant
        Horn clauses, conjuncts them as the new refinement.
        """
        info = self._vars[name]
        pred_parts: List[LiquidExpr] = []

        # gather predicates from predecessors
        for pred_name in info.predecessors:
            pred_info = self._vars.get(pred_name)
            if pred_info is not None:
                renamed = pred_info.current_type.predicate.substitute(
                    {pred_info.current_type.var: LiquidVar(info.current_type.var)}
                )
                pred_parts.append(renamed)

        # gather heads of Horn clauses whose body is satisfied
        for clause in self._horn_clauses:
            head_fv = clause.head.free_vars()
            if info.current_type.var in head_fv or name in head_fv:
                body_ok = self._check_body(clause, name)
                if body_ok:
                    head_mapped = clause.head.substitute(
                        {name: LiquidVar(info.current_type.var)}
                    )
                    pred_parts.append(head_mapped)

        if not pred_parts:
            return None

        new_pred = liquid_and(pred_parts)
        return LiquidType(
            var=info.current_type.var,
            base=info.current_type.base,
            predicate=new_pred,
        )

    def _check_body(self, clause: HornClause, target_var: str) -> bool:
        """Heuristic check whether clause body is satisfiable given current solution."""
        for b in clause.body:
            fv = b.free_vars()
            for v in fv:
                if v in self._vars and self._vars[v].current_type.predicate == FALSE:
                    return False
        return True

    def _widen(self, old: LiquidType, new: LiquidType) -> LiquidType:
        """Widening: drop the new predicate and keep only the base type.

        This guarantees termination at the cost of precision.
        """
        return LiquidType(var=old.var, base=old.base, predicate=TRUE)

    @staticmethod
    def _types_equal(a: LiquidType, b: LiquidType) -> bool:
        """Syntactic equality of two liquid types (conservative)."""
        return a.var == b.var and a.base == b.base and a.predicate == b.predicate
