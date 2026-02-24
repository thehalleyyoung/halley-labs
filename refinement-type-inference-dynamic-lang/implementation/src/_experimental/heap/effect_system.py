"""
Effect system for tracking side effects in Python programs.

Part of a refinement type inference system. Provides fine-grained
tracking of heap reads/writes, I/O, exceptions, allocations, and
external calls so that downstream passes can reason about purity,
commutativity, and safe reordering of expressions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from .class_hierarchy import ClassHierarchyAnalyzer, ClassInfo, MethodInfo


# ===================================================================
# Enums
# ===================================================================

class Effect(Enum):
    """Individual side-effect categories."""

    PURE = auto()
    READ_HEAP = auto()
    WRITE_HEAP = auto()
    READ_IO = auto()
    WRITE_IO = auto()
    RAISE = auto()
    ALLOCATE = auto()
    CALL_EXTERNAL = auto()
    READ_GLOBAL = auto()
    WRITE_GLOBAL = auto()
    READ_NONLOCAL = auto()
    WRITE_NONLOCAL = auto()
    READ_CLOSURE = auto()
    WRITE_CLOSURE = auto()


# ===================================================================
# Data classes
# ===================================================================

@dataclass(frozen=True)
class EffectSet:
    """An immutable set of effects together with detailed location info."""

    effects: FrozenSet[Effect] = frozenset()
    read_locations: FrozenSet[str] = frozenset()
    write_locations: FrozenSet[str] = frozenset()
    raised_exceptions: FrozenSet[str] = frozenset()
    allocated_types: FrozenSet[str] = frozenset()
    called_functions: FrozenSet[str] = frozenset()
    io_operations: FrozenSet[str] = frozenset()

    # -- combinators ------------------------------------------------

    def union(self, other: EffectSet) -> EffectSet:
        """Return the union (join) of two effect sets."""
        combined_effects = self.effects | other.effects
        # PURE is only meaningful when nothing else is present
        if len(combined_effects) > 1 and Effect.PURE in combined_effects:
            combined_effects = combined_effects - {Effect.PURE}
        return EffectSet(
            effects=combined_effects,
            read_locations=self.read_locations | other.read_locations,
            write_locations=self.write_locations | other.write_locations,
            raised_exceptions=self.raised_exceptions | other.raised_exceptions,
            allocated_types=self.allocated_types | other.allocated_types,
            called_functions=self.called_functions | other.called_functions,
            io_operations=self.io_operations | other.io_operations,
        )

    def intersection(self, other: EffectSet) -> EffectSet:
        """Return the intersection (meet) of two effect sets."""
        return EffectSet(
            effects=self.effects & other.effects,
            read_locations=self.read_locations & other.read_locations,
            write_locations=self.write_locations & other.write_locations,
            raised_exceptions=self.raised_exceptions & other.raised_exceptions,
            allocated_types=self.allocated_types & other.allocated_types,
            called_functions=self.called_functions & other.called_functions,
            io_operations=self.io_operations & other.io_operations,
        )

    # -- predicates -------------------------------------------------

    def is_pure(self) -> bool:
        """True when no observable side effects exist."""
        non_pure = self.effects - {Effect.PURE}
        return len(non_pure) == 0

    def is_read_only(self) -> bool:
        """True when no write or I/O-write effects exist."""
        write_effects = {
            Effect.WRITE_HEAP,
            Effect.WRITE_IO,
            Effect.WRITE_GLOBAL,
            Effect.WRITE_NONLOCAL,
            Effect.WRITE_CLOSURE,
        }
        return not (self.effects & write_effects)

    def is_local(self) -> bool:
        """True when effects do not escape the current stack frame."""
        non_local = {
            Effect.READ_HEAP,
            Effect.WRITE_HEAP,
            Effect.READ_IO,
            Effect.WRITE_IO,
            Effect.CALL_EXTERNAL,
            Effect.READ_GLOBAL,
            Effect.WRITE_GLOBAL,
            Effect.READ_NONLOCAL,
            Effect.WRITE_NONLOCAL,
            Effect.READ_CLOSURE,
            Effect.WRITE_CLOSURE,
        }
        return not (self.effects & non_local)

    def subsumes(self, other: EffectSet) -> bool:
        """True when *self* is a superset of *other* in every dimension."""
        return (
            other.effects <= self.effects
            and other.read_locations <= self.read_locations
            and other.write_locations <= self.write_locations
            and other.raised_exceptions <= self.raised_exceptions
            and other.allocated_types <= self.allocated_types
            and other.called_functions <= self.called_functions
            and other.io_operations <= self.io_operations
        )

    # -- display ----------------------------------------------------

    def pretty(self) -> str:
        """Human-readable summary."""
        if self.is_pure():
            return "PURE"
        parts: list[str] = []
        for eff in sorted(self.effects, key=lambda e: e.name):
            if eff is Effect.PURE:
                continue
            parts.append(eff.name)
        extras: list[str] = []
        if self.read_locations:
            extras.append(f"reads={{{', '.join(sorted(self.read_locations))}}}")
        if self.write_locations:
            extras.append(f"writes={{{', '.join(sorted(self.write_locations))}}}")
        if self.raised_exceptions:
            extras.append(f"raises={{{', '.join(sorted(self.raised_exceptions))}}}")
        if self.allocated_types:
            extras.append(f"allocs={{{', '.join(sorted(self.allocated_types))}}}")
        if self.called_functions:
            extras.append(f"calls={{{', '.join(sorted(self.called_functions))}}}")
        if self.io_operations:
            extras.append(f"io={{{', '.join(sorted(self.io_operations))}}}")
        result = " | ".join(parts)
        if extras:
            result += " [" + "; ".join(extras) + "]"
        return result


@dataclass
class FunctionEffectSummary:
    """Aggregated effect summary for a whole function."""

    function_name: str
    effects: EffectSet
    param_effects: Dict[str, EffectSet] = field(default_factory=dict)
    return_effects: EffectSet = field(default_factory=EffectSet)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class ExpressionEffect:
    """Effect of evaluating a single expression."""

    expr_description: str
    effects: EffectSet


# ===================================================================
# Constants
# ===================================================================

_PURE = EffectSet(effects=frozenset({Effect.PURE}))

_READ_HEAP = EffectSet(effects=frozenset({Effect.READ_HEAP}))

_WRITE_HEAP = EffectSet(effects=frozenset({Effect.WRITE_HEAP}))

_ALLOCATE = EffectSet(effects=frozenset({Effect.ALLOCATE}))

_RAISE = EffectSet(effects=frozenset({Effect.RAISE}))


# ===================================================================
# EffectAnalyzer
# ===================================================================

class EffectAnalyzer:
    """Walk Python AST nodes and compute fine-grained effect sets."""

    def __init__(self, hierarchy: ClassHierarchyAnalyzer) -> None:
        self.hierarchy = hierarchy
        self._builtin_effects: Dict[str, EffectSet] = self._init_builtin_effects()
        # Track scope information during analysis
        self._global_names: Set[str] = set()
        self._nonlocal_names: Set[str] = set()

    # ------------------------------------------------------------------
    # Builtin effect table
    # ------------------------------------------------------------------

    def _init_builtin_effects(self) -> Dict[str, EffectSet]:
        """Return known effects for Python built-in functions."""
        pure = _PURE
        read_heap = _READ_HEAP
        write_io = EffectSet(
            effects=frozenset({Effect.WRITE_IO}),
            io_operations=frozenset({"stdout"}),
        )
        read_io = EffectSet(
            effects=frozenset({Effect.READ_IO}),
            io_operations=frozenset({"stdin"}),
        )
        read_write_io = EffectSet(
            effects=frozenset({Effect.READ_IO, Effect.WRITE_IO}),
            io_operations=frozenset({"stdin", "stdout"}),
        )
        alloc = _ALLOCATE
        alloc_pure = EffectSet(effects=frozenset({Effect.PURE, Effect.ALLOCATE}))
        read_heap_alloc = EffectSet(effects=frozenset({Effect.READ_HEAP, Effect.ALLOCATE}))

        # All effects (for exec/eval)
        all_effects = EffectSet(
            effects=frozenset(Effect),
            io_operations=frozenset({"exec/eval"}),
        )

        iter_effects = EffectSet(
            effects=frozenset({Effect.READ_HEAP, Effect.RAISE}),
            raised_exceptions=frozenset({"StopIteration"}),
        )

        import_effects = EffectSet(
            effects=frozenset({Effect.READ_IO, Effect.CALL_EXTERNAL}),
            io_operations=frozenset({"import"}),
        )

        return {
            # I/O
            "print": write_io,
            "input": read_write_io,
            "open": EffectSet(
                effects=frozenset({Effect.READ_IO, Effect.WRITE_IO, Effect.ALLOCATE}),
                io_operations=frozenset({"file"}),
                allocated_types=frozenset({"file"}),
            ),
            # Pure conversions
            "int": pure,
            "str": pure,
            "float": pure,
            "bool": pure,
            "complex": pure,
            "bytes": pure,
            "bytearray": alloc,
            "memoryview": alloc,
            "chr": pure,
            "ord": pure,
            "hex": pure,
            "oct": pure,
            "bin": pure,
            "ascii": pure,
            "repr": EffectSet(effects=frozenset({Effect.READ_HEAP})),
            "format": EffectSet(effects=frozenset({Effect.READ_HEAP})),
            # Allocating constructors
            "list": EffectSet(effects=frozenset({Effect.ALLOCATE}), allocated_types=frozenset({"list"})),
            "dict": EffectSet(effects=frozenset({Effect.ALLOCATE}), allocated_types=frozenset({"dict"})),
            "set": EffectSet(effects=frozenset({Effect.ALLOCATE}), allocated_types=frozenset({"set"})),
            "frozenset": EffectSet(effects=frozenset({Effect.ALLOCATE}), allocated_types=frozenset({"frozenset"})),
            "tuple": EffectSet(effects=frozenset({Effect.ALLOCATE}), allocated_types=frozenset({"tuple"})),
            "object": EffectSet(effects=frozenset({Effect.ALLOCATE}), allocated_types=frozenset({"object"})),
            # Sorting / ranges
            "sorted": EffectSet(
                effects=frozenset({Effect.READ_HEAP, Effect.ALLOCATE}),
                allocated_types=frozenset({"list"}),
            ),
            "reversed": read_heap_alloc,
            "range": EffectSet(
                effects=frozenset({Effect.PURE, Effect.ALLOCATE}),
                allocated_types=frozenset({"range"}),
            ),
            # Introspection (pure / read-heap)
            "len": EffectSet(effects=frozenset({Effect.READ_HEAP})),
            "isinstance": read_heap,
            "issubclass": read_heap,
            "hasattr": read_heap,
            "getattr": read_heap,
            "setattr": EffectSet(effects=frozenset({Effect.WRITE_HEAP})),
            "delattr": EffectSet(effects=frozenset({Effect.WRITE_HEAP})),
            "id": pure,
            "hash": pure,
            "type": pure,
            "callable": read_heap,
            "super": read_heap,
            # Iterators
            "iter": iter_effects,
            "next": iter_effects,
            # Aggregation builtins
            "all": read_heap,
            "any": read_heap,
            "sum": read_heap,
            "min": read_heap,
            "max": read_heap,
            "abs": pure,
            "round": pure,
            "pow": pure,
            "divmod": pure,
            # Higher-order
            "map": read_heap_alloc,
            "filter": read_heap_alloc,
            "zip": read_heap_alloc,
            "enumerate": read_heap_alloc,
            # Dynamic execution
            "exec": all_effects,
            "eval": all_effects,
            "compile": EffectSet(effects=frozenset({Effect.ALLOCATE})),
            # Scope inspection
            "globals": EffectSet(effects=frozenset({Effect.READ_GLOBAL})),
            "locals": EffectSet(effects=frozenset({Effect.READ_HEAP})),
            "vars": read_heap,
            "dir": read_heap,
            # Import
            "__import__": import_effects,
            # Misc
            "staticmethod": pure,
            "classmethod": pure,
            "property": pure,
            "breakpoint": EffectSet(
                effects=frozenset({Effect.READ_IO, Effect.WRITE_IO}),
                io_operations=frozenset({"debugger"}),
            ),
            "exit": EffectSet(
                effects=frozenset({Effect.RAISE}),
                raised_exceptions=frozenset({"SystemExit"}),
            ),
            "quit": EffectSet(
                effects=frozenset({Effect.RAISE}),
                raised_exceptions=frozenset({"SystemExit"}),
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_function(self, func: ast.FunctionDef) -> FunctionEffectSummary:
        """Compute an effect summary for *func* by walking its body."""
        saved_globals = self._global_names.copy()
        saved_nonlocals = self._nonlocal_names.copy()
        self._global_names = set()
        self._nonlocal_names = set()

        body_effects = self._empty_effects()
        for stmt in func.body:
            body_effects = body_effects.union(self.analyze_statement(stmt))

        # Per-parameter effects: attribute / subscript access on params
        param_names = {
            arg.arg for arg in func.args.args + func.args.posonlyargs + func.args.kwonlyargs
        }
        if func.args.vararg:
            param_names.add(func.args.vararg.arg)
        if func.args.kwarg:
            param_names.add(func.args.kwarg.arg)

        param_effects: Dict[str, EffectSet] = {}
        for name in param_names:
            pe = self._collect_param_effects(func.body, name)
            if not pe.is_pure():
                param_effects[name] = pe

        # Return effects
        return_effects = self._collect_return_effects(func.body)

        # Preconditions / postconditions from docstring (simple heuristic)
        preconditions: List[str] = []
        postconditions: List[str] = []
        docstring = ast.get_docstring(func)
        if docstring:
            for line in docstring.splitlines():
                stripped = line.strip().lower()
                if stripped.startswith("requires:") or stripped.startswith("pre:"):
                    preconditions.append(line.strip())
                elif stripped.startswith("ensures:") or stripped.startswith("post:"):
                    postconditions.append(line.strip())

        self._global_names = saved_globals
        self._nonlocal_names = saved_nonlocals

        return FunctionEffectSummary(
            function_name=func.name,
            effects=body_effects,
            param_effects=param_effects,
            return_effects=return_effects,
            preconditions=preconditions,
            postconditions=postconditions,
        )

    def analyze_expression(self, expr: ast.expr) -> EffectSet:
        """Compute effects for a single expression node."""
        dispatch = {
            ast.Name: self._analyze_name,
            ast.Attribute: self._analyze_attribute,
            ast.Subscript: self._analyze_subscript,
            ast.Call: self._analyze_call,
            ast.BinOp: self._analyze_binop,
            ast.Compare: self._analyze_compare,
            ast.BoolOp: self._analyze_boolop,
            ast.IfExp: self._analyze_ifexp,
            ast.Lambda: self._analyze_lambda,
            ast.ListComp: self._analyze_listcomp,
            ast.DictComp: self._analyze_dictcomp,
            ast.SetComp: self._analyze_setcomp,
            ast.GeneratorExp: self._analyze_genexp,
            ast.Await: self._analyze_await,
            ast.Yield: self._analyze_yield,
            ast.YieldFrom: self._analyze_yield,
            ast.JoinedStr: self._analyze_fstring,
            ast.UnaryOp: self._analyze_unaryop,
            ast.Starred: self._analyze_starred,
        }
        handler = dispatch.get(type(expr))
        if handler is not None:
            return handler(expr)  # type: ignore[arg-type]

        # Constants, NameConstant, Num, Str, FormattedValue …
        if isinstance(expr, ast.Constant):
            return _PURE

        # List / Tuple / Set / Dict literals
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            eff = EffectSet(
                effects=frozenset({Effect.ALLOCATE}),
                allocated_types=frozenset({type(expr).__name__.lower()}),
            )
            for elt in expr.elts:
                eff = eff.union(self.analyze_expression(elt))
            return eff

        if isinstance(expr, ast.Dict):
            eff = EffectSet(
                effects=frozenset({Effect.ALLOCATE}),
                allocated_types=frozenset({"dict"}),
            )
            for k, v in zip(expr.keys, expr.values):
                if k is not None:
                    eff = eff.union(self.analyze_expression(k))
                eff = eff.union(self.analyze_expression(v))
            return eff

        if isinstance(expr, ast.FormattedValue):
            eff = self.analyze_expression(expr.value)
            if expr.format_spec:
                eff = eff.union(self.analyze_expression(expr.format_spec))
            return eff.union(EffectSet(effects=frozenset({Effect.READ_HEAP})))

        if isinstance(expr, ast.Slice):
            eff = _PURE
            if expr.lower:
                eff = eff.union(self.analyze_expression(expr.lower))
            if expr.upper:
                eff = eff.union(self.analyze_expression(expr.upper))
            if expr.step:
                eff = eff.union(self.analyze_expression(expr.step))
            return eff

        return _PURE

    def analyze_statement(self, stmt: ast.stmt) -> EffectSet:
        """Compute effects of a single statement."""
        dispatch = {
            ast.Assign: self._analyze_assign,
            ast.AugAssign: self._analyze_augassign,
            ast.AnnAssign: self._analyze_annassign,
            ast.For: self._analyze_for,
            ast.AsyncFor: self._analyze_for,
            ast.While: self._analyze_while,
            ast.If: self._analyze_if,
            ast.With: self._analyze_with,
            ast.AsyncWith: self._analyze_with,
            ast.Try: self._analyze_try,
            ast.Raise: self._analyze_raise,
            ast.Return: self._analyze_return,
            ast.Delete: self._analyze_delete,
            ast.Global: self._analyze_global,
            ast.Nonlocal: self._analyze_nonlocal,
            ast.Assert: self._analyze_assert,
            ast.Import: self._analyze_import,
            ast.ImportFrom: self._analyze_import,
            ast.Pass: lambda _: _PURE,
            ast.Break: lambda _: _PURE,
            ast.Continue: lambda _: _PURE,
        }
        handler = dispatch.get(type(stmt))
        if handler is not None:
            return handler(stmt)  # type: ignore[arg-type]

        # Expression statement
        if isinstance(stmt, ast.Expr):
            return self.analyze_expression(stmt.value)

        # FunctionDef / AsyncFunctionDef / ClassDef — defining them allocates
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            eff = EffectSet(
                effects=frozenset({Effect.ALLOCATE}),
                allocated_types=frozenset({"function"}),
            )
            for decorator in stmt.decorator_list:
                eff = eff.union(self.analyze_expression(decorator))
            return eff

        if isinstance(stmt, ast.ClassDef):
            eff = EffectSet(
                effects=frozenset({Effect.ALLOCATE}),
                allocated_types=frozenset({"class"}),
            )
            for base in stmt.bases:
                eff = eff.union(self.analyze_expression(base))
            for decorator in stmt.decorator_list:
                eff = eff.union(self.analyze_expression(decorator))
            return eff

        # TryStar (Python 3.11+)
        if hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar):  # type: ignore[attr-defined]
            return self._analyze_try(stmt)  # type: ignore[arg-type]

        return _PURE

    # -- convenience wrappers ----------------------------------------

    def is_pure(self, func: ast.FunctionDef) -> bool:
        """Check if *func* has no side effects."""
        return self.analyze_function(func).effects.is_pure()

    def is_read_only(self, func: ast.FunctionDef) -> bool:
        """Check if *func* only reads (no writes)."""
        return self.analyze_function(func).effects.is_read_only()

    def get_read_set(self, func: ast.FunctionDef) -> Set[str]:
        """Return the set of heap locations read."""
        return set(self.analyze_function(func).effects.read_locations)

    def get_write_set(self, func: ast.FunctionDef) -> Set[str]:
        """Return the set of heap locations written."""
        return set(self.analyze_function(func).effects.write_locations)

    def get_raised_exceptions(self, func: ast.FunctionDef) -> Set[str]:
        """Return exception types that *func* may raise."""
        return set(self.analyze_function(func).effects.raised_exceptions)

    def can_commute(self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> bool:
        """True when *func1* and *func2* can be safely reordered.

        Two functions commute when their write sets are disjoint from
        each other's read and write sets, and neither performs I/O.
        """
        s1 = self.analyze_function(func1).effects
        s2 = self.analyze_function(func2).effects

        io_effects = {Effect.READ_IO, Effect.WRITE_IO}
        if s1.effects & io_effects or s2.effects & io_effects:
            return False

        if s1.write_locations & s2.read_locations:
            return False
        if s1.write_locations & s2.write_locations:
            return False
        if s2.write_locations & s1.read_locations:
            return False

        # If either calls unknown externals we can't guarantee commutativity
        if Effect.CALL_EXTERNAL in s1.effects or Effect.CALL_EXTERNAL in s2.effects:
            return False

        return True

    # ------------------------------------------------------------------
    # Expression analyzers
    # ------------------------------------------------------------------

    def _analyze_name(self, node: ast.Name) -> EffectSet:
        """Variable read."""
        if node.id in self._global_names:
            return EffectSet(
                effects=frozenset({Effect.READ_GLOBAL}),
                read_locations=frozenset({f"global:{node.id}"}),
            )
        if node.id in self._nonlocal_names:
            return EffectSet(
                effects=frozenset({Effect.READ_NONLOCAL}),
                read_locations=frozenset({f"nonlocal:{node.id}"}),
            )
        return _PURE

    def _analyze_attribute(self, node: ast.Attribute) -> EffectSet:
        """Attribute access reads the heap."""
        value_eff = self.analyze_expression(node.value)
        loc = self._location_of_attribute(node)
        return value_eff.union(EffectSet(
            effects=frozenset({Effect.READ_HEAP}),
            read_locations=frozenset({loc}),
        ))

    def _analyze_subscript(self, node: ast.Subscript) -> EffectSet:
        """Subscript access (``x[i]``) reads the heap."""
        value_eff = self.analyze_expression(node.value)
        slice_eff = self.analyze_expression(node.slice)
        return value_eff.union(slice_eff).union(_READ_HEAP)

    def _analyze_call(self, node: ast.Call) -> EffectSet:
        """Resolve call target and look up its effects."""
        target = self._get_call_target(node)

        # Effects from evaluating arguments
        arg_eff = self._empty_effects()
        for arg in node.args:
            arg_eff = arg_eff.union(self.analyze_expression(arg))
        for kw in node.keywords:
            arg_eff = arg_eff.union(self.analyze_expression(kw.value))

        # Effects from evaluating the callee expression itself
        callee_eff = self.analyze_expression(node.func)

        if target and target in self._builtin_effects:
            call_eff = self._builtin_effects[target]
            return self._merge_effects(arg_eff, callee_eff, EffectSet(
                effects=call_eff.effects,
                read_locations=call_eff.read_locations,
                write_locations=call_eff.write_locations,
                raised_exceptions=call_eff.raised_exceptions,
                allocated_types=call_eff.allocated_types,
                called_functions=frozenset({target}),
                io_operations=call_eff.io_operations,
            ))

        # Method calls on known types
        if target and "." in target:
            parts = target.rsplit(".", 1)
            method_eff = self._method_effects(parts[0], parts[1])
            if method_eff is not None:
                return self._merge_effects(arg_eff, callee_eff, method_eff)

        # Unknown call — conservative
        fname = target or "<unknown>"
        return self._merge_effects(arg_eff, callee_eff, EffectSet(
            effects=frozenset({Effect.CALL_EXTERNAL, Effect.READ_HEAP, Effect.WRITE_HEAP}),
            called_functions=frozenset({fname}),
        ))

    def _analyze_binop(self, node: ast.BinOp) -> EffectSet:
        """Binary operators are pure for primitive types."""
        left = self.analyze_expression(node.left)
        right = self.analyze_expression(node.right)
        return left.union(right)

    def _analyze_unaryop(self, node: ast.UnaryOp) -> EffectSet:
        """Unary operators are typically pure."""
        return self.analyze_expression(node.operand)

    def _analyze_compare(self, node: ast.Compare) -> EffectSet:
        """Comparisons may invoke __eq__, __lt__, etc."""
        eff = self.analyze_expression(node.left)
        for comparator in node.comparators:
            eff = eff.union(self.analyze_expression(comparator))
        # Comparisons on user objects can read heap (__eq__ etc.)
        return eff.union(_READ_HEAP)

    def _analyze_boolop(self, node: ast.BoolOp) -> EffectSet:
        """``and`` / ``or`` — short-circuit: union of all operand effects."""
        eff = self._empty_effects()
        for value in node.values:
            eff = eff.union(self.analyze_expression(value))
        return eff

    def _analyze_ifexp(self, node: ast.IfExp) -> EffectSet:
        """Ternary expression ``a if cond else b``."""
        return self._merge_effects(
            self.analyze_expression(node.test),
            self.analyze_expression(node.body),
            self.analyze_expression(node.orelse),
        )

    def _analyze_lambda(self, node: ast.Lambda) -> EffectSet:
        """Lambda creates a closure object."""
        return EffectSet(
            effects=frozenset({Effect.ALLOCATE}),
            allocated_types=frozenset({"function"}),
        )

    def _analyze_listcomp(self, node: ast.ListComp) -> EffectSet:
        """List comprehension allocates and iterates."""
        return self._analyze_comprehension(node.generators, node.elt, "list")

    def _analyze_dictcomp(self, node: ast.DictComp) -> EffectSet:
        """Dict comprehension."""
        eff = self._analyze_generators(node.generators)
        eff = eff.union(self.analyze_expression(node.key))
        eff = eff.union(self.analyze_expression(node.value))
        return eff.union(EffectSet(
            effects=frozenset({Effect.ALLOCATE}),
            allocated_types=frozenset({"dict"}),
        ))

    def _analyze_setcomp(self, node: ast.SetComp) -> EffectSet:
        """Set comprehension."""
        return self._analyze_comprehension(node.generators, node.elt, "set")

    def _analyze_genexp(self, node: ast.GeneratorExp) -> EffectSet:
        """Generator expression allocates a generator."""
        eff = self._analyze_generators(node.generators)
        eff = eff.union(self.analyze_expression(node.elt))
        return eff.union(EffectSet(
            effects=frozenset({Effect.ALLOCATE}),
            allocated_types=frozenset({"generator"}),
        ))

    def _analyze_await(self, node: ast.Await) -> EffectSet:
        """``await`` may perform I/O and read/write heap."""
        inner = self.analyze_expression(node.value)
        return inner.union(EffectSet(
            effects=frozenset({Effect.READ_IO, Effect.WRITE_IO, Effect.READ_HEAP}),
            io_operations=frozenset({"await"}),
        ))

    def _analyze_yield(self, node: ast.Yield | ast.YieldFrom) -> EffectSet:
        """``yield`` / ``yield from`` suspends and resumes the generator."""
        eff = EffectSet(effects=frozenset({Effect.READ_HEAP, Effect.WRITE_HEAP}))
        if isinstance(node, ast.Yield) and node.value:
            eff = eff.union(self.analyze_expression(node.value))
        elif isinstance(node, ast.YieldFrom) and node.value:
            eff = eff.union(self.analyze_expression(node.value))
        return eff

    def _analyze_fstring(self, node: ast.JoinedStr) -> EffectSet:
        """f-string may call ``__format__`` on interpolated values."""
        eff = EffectSet(effects=frozenset({Effect.READ_HEAP}))
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                eff = eff.union(self.analyze_expression(value.value))
                if value.format_spec:
                    eff = eff.union(self.analyze_expression(value.format_spec))
            else:
                eff = eff.union(self.analyze_expression(value))
        return eff

    def _analyze_starred(self, node: ast.Starred) -> EffectSet:
        """Starred expression unpacks an iterable."""
        return self.analyze_expression(node.value).union(_READ_HEAP)

    # ------------------------------------------------------------------
    # Statement analyzers
    # ------------------------------------------------------------------

    def _analyze_assign(self, node: ast.Assign) -> EffectSet:
        """Assignment statement."""
        value_eff = self.analyze_expression(node.value)
        target_eff = self._empty_effects()
        for target in node.targets:
            target_eff = target_eff.union(self._analyze_write_target(target))
        return value_eff.union(target_eff)

    def _analyze_annassign(self, node: ast.AnnAssign) -> EffectSet:
        """Annotated assignment."""
        eff = self._empty_effects()
        if node.value:
            eff = eff.union(self.analyze_expression(node.value))
        if node.target:
            eff = eff.union(self._analyze_write_target(node.target))
        return eff

    def _analyze_augassign(self, node: ast.AugAssign) -> EffectSet:
        """Augmented assignment (``x += …``)."""
        value_eff = self.analyze_expression(node.value)
        target_read_eff = self.analyze_expression(node.target)
        target_write_eff = self._analyze_write_target(node.target)
        return self._merge_effects(value_eff, target_read_eff, target_write_eff)

    def _analyze_for(self, node: ast.For | ast.AsyncFor) -> EffectSet:
        """For loop iterates (reads heap) and runs a body."""
        iter_eff = self.analyze_expression(node.iter)
        target_eff = self._analyze_write_target(node.target)
        body_eff = self._analyze_body(node.body)
        orelse_eff = self._analyze_body(node.orelse)
        loop_eff = EffectSet(
            effects=frozenset({Effect.READ_HEAP}),
        )
        if isinstance(node, ast.AsyncFor):
            loop_eff = loop_eff.union(EffectSet(
                effects=frozenset({Effect.READ_IO}),
                io_operations=frozenset({"async_iter"}),
            ))
        return self._merge_effects(iter_eff, target_eff, body_eff, orelse_eff, loop_eff)

    def _analyze_while(self, node: ast.While) -> EffectSet:
        """While loop."""
        test_eff = self.analyze_expression(node.test)
        body_eff = self._analyze_body(node.body)
        orelse_eff = self._analyze_body(node.orelse)
        return self._merge_effects(test_eff, body_eff, orelse_eff)

    def _analyze_if(self, node: ast.If) -> EffectSet:
        """If statement — union of both branches."""
        test_eff = self.analyze_expression(node.test)
        body_eff = self._analyze_body(node.body)
        orelse_eff = self._analyze_body(node.orelse)
        return self._merge_effects(test_eff, body_eff, orelse_eff)

    def _analyze_with(self, node: ast.With | ast.AsyncWith) -> EffectSet:
        """``with`` invokes ``__enter__`` and ``__exit__``."""
        eff = EffectSet(
            effects=frozenset({Effect.READ_HEAP, Effect.WRITE_HEAP, Effect.ALLOCATE}),
        )
        if isinstance(node, ast.AsyncWith):
            eff = eff.union(EffectSet(
                effects=frozenset({Effect.READ_IO, Effect.WRITE_IO}),
                io_operations=frozenset({"async_context_manager"}),
            ))
        for item in node.items:
            eff = eff.union(self.analyze_expression(item.context_expr))
            if item.optional_vars:
                eff = eff.union(self._analyze_write_target(item.optional_vars))
        eff = eff.union(self._analyze_body(node.body))
        return eff

    def _analyze_try(self, node: ast.Try) -> EffectSet:
        """Try/except/finally."""
        body_eff = self._analyze_body(node.body)
        handler_eff = self._empty_effects()
        for handler in node.handlers:
            h_eff = self._analyze_body(handler.body)
            if handler.type:
                exc_name = self._exception_name(handler.type)
                h_eff = h_eff.union(EffectSet(
                    effects=frozenset({Effect.RAISE}),
                    raised_exceptions=frozenset({exc_name}),
                ))
            handler_eff = handler_eff.union(h_eff)
        orelse_eff = self._analyze_body(node.orelse)
        finally_eff = self._analyze_body(node.finalbody)
        return self._merge_effects(body_eff, handler_eff, orelse_eff, finally_eff)

    def _analyze_raise(self, node: ast.Raise) -> EffectSet:
        """Raise statement."""
        eff = EffectSet(effects=frozenset({Effect.RAISE}))
        if node.exc:
            exc_eff = self.analyze_expression(node.exc)
            exc_name = self._exception_name(node.exc)
            eff = eff.union(exc_eff).union(EffectSet(
                raised_exceptions=frozenset({exc_name}),
            ))
        if node.cause:
            eff = eff.union(self.analyze_expression(node.cause))
        return eff

    def _analyze_return(self, node: ast.Return) -> EffectSet:
        """Return statement — effects come from the returned expression."""
        if node.value:
            return self.analyze_expression(node.value)
        return _PURE

    def _analyze_delete(self, node: ast.Delete) -> EffectSet:
        """``del`` writes the heap."""
        eff = self._empty_effects()
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                loc = self._location_of_attribute(target)
                eff = eff.union(self.analyze_expression(target.value))
                eff = eff.union(EffectSet(
                    effects=frozenset({Effect.WRITE_HEAP}),
                    write_locations=frozenset({loc}),
                ))
            elif isinstance(target, ast.Subscript):
                eff = eff.union(self.analyze_expression(target.value))
                eff = eff.union(self.analyze_expression(target.slice))
                eff = eff.union(_WRITE_HEAP)
            elif isinstance(target, ast.Name):
                if target.id in self._global_names:
                    eff = eff.union(EffectSet(
                        effects=frozenset({Effect.WRITE_GLOBAL}),
                        write_locations=frozenset({f"global:{target.id}"}),
                    ))
                elif target.id in self._nonlocal_names:
                    eff = eff.union(EffectSet(
                        effects=frozenset({Effect.WRITE_NONLOCAL}),
                        write_locations=frozenset({f"nonlocal:{target.id}"}),
                    ))
            else:
                eff = eff.union(_WRITE_HEAP)
        return eff

    def _analyze_global(self, node: ast.Global) -> EffectSet:
        """``global`` declaration marks names as global-scope."""
        for name in node.names:
            self._global_names.add(name)
        return EffectSet(
            effects=frozenset({Effect.READ_GLOBAL}),
            read_locations=frozenset(f"global:{n}" for n in node.names),
        )

    def _analyze_nonlocal(self, node: ast.Nonlocal) -> EffectSet:
        """``nonlocal`` declaration marks names as enclosing-scope."""
        for name in node.names:
            self._nonlocal_names.add(name)
        return EffectSet(
            effects=frozenset({Effect.READ_NONLOCAL}),
            read_locations=frozenset(f"nonlocal:{n}" for n in node.names),
        )

    def _analyze_assert(self, node: ast.Assert) -> EffectSet:
        """``assert`` may raise ``AssertionError``."""
        eff = self.analyze_expression(node.test)
        if node.msg:
            eff = eff.union(self.analyze_expression(node.msg))
        return eff.union(EffectSet(
            effects=frozenset({Effect.RAISE}),
            raised_exceptions=frozenset({"AssertionError"}),
        ))

    def _analyze_import(self, node: ast.Import | ast.ImportFrom) -> EffectSet:
        """Import statements perform I/O and external calls."""
        return EffectSet(
            effects=frozenset({Effect.READ_IO, Effect.CALL_EXTERNAL, Effect.ALLOCATE}),
            io_operations=frozenset({"import"}),
            allocated_types=frozenset({"module"}),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_call_target(self, node: ast.Call) -> Optional[str]:
        """Resolve the function being called to a name string, if possible."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = self._expr_to_str(func.value)
            if base:
                return f"{base}.{func.attr}"
            return func.attr
        return None

    def _merge_effects(self, *effect_sets: EffectSet) -> EffectSet:
        """Compute the union of arbitrarily many effect sets."""
        result = self._empty_effects()
        for es in effect_sets:
            result = result.union(es)
        return result

    def _empty_effects(self) -> EffectSet:
        """Create an empty (pure) effect set."""
        return _PURE

    # -- write-target analysis ----------------------------------------

    def _analyze_write_target(self, target: ast.expr) -> EffectSet:
        """Determine the effects of writing to *target*."""
        if isinstance(target, ast.Name):
            if target.id in self._global_names:
                return EffectSet(
                    effects=frozenset({Effect.WRITE_GLOBAL}),
                    write_locations=frozenset({f"global:{target.id}"}),
                )
            if target.id in self._nonlocal_names:
                return EffectSet(
                    effects=frozenset({Effect.WRITE_NONLOCAL}),
                    write_locations=frozenset({f"nonlocal:{target.id}"}),
                )
            return _PURE  # local variable write — no heap effect

        if isinstance(target, ast.Attribute):
            value_eff = self.analyze_expression(target.value)
            loc = self._location_of_attribute(target)
            return value_eff.union(EffectSet(
                effects=frozenset({Effect.WRITE_HEAP}),
                write_locations=frozenset({loc}),
            ))

        if isinstance(target, ast.Subscript):
            value_eff = self.analyze_expression(target.value)
            slice_eff = self.analyze_expression(target.slice)
            return value_eff.union(slice_eff).union(_WRITE_HEAP)

        if isinstance(target, ast.Starred):
            return self._analyze_write_target(target.value)

        if isinstance(target, (ast.Tuple, ast.List)):
            eff = self._empty_effects()
            for elt in target.elts:
                eff = eff.union(self._analyze_write_target(elt))
            return eff

        return _WRITE_HEAP

    # -- comprehension helpers ----------------------------------------

    def _analyze_comprehension(
        self,
        generators: list[ast.comprehension],
        elt: ast.expr,
        alloc_type: str,
    ) -> EffectSet:
        """Shared logic for list/set comprehension analysis."""
        eff = self._analyze_generators(generators)
        eff = eff.union(self.analyze_expression(elt))
        return eff.union(EffectSet(
            effects=frozenset({Effect.ALLOCATE}),
            allocated_types=frozenset({alloc_type}),
        ))

    def _analyze_generators(
        self, generators: list[ast.comprehension]
    ) -> EffectSet:
        """Analyze comprehension generators (``for … in … if …``)."""
        eff = self._empty_effects()
        for gen in generators:
            eff = eff.union(self.analyze_expression(gen.iter))
            eff = eff.union(self._analyze_write_target(gen.target))
            for if_clause in gen.ifs:
                eff = eff.union(self.analyze_expression(if_clause))
        return eff.union(_READ_HEAP)  # iteration reads heap

    # -- body / block helpers ----------------------------------------

    def _analyze_body(self, stmts: list[ast.stmt]) -> EffectSet:
        """Analyze a sequence of statements."""
        eff = self._empty_effects()
        for stmt in stmts:
            eff = eff.union(self.analyze_statement(stmt))
        return eff

    # -- per-parameter effect collection ------------------------------

    def _collect_param_effects(
        self, body: list[ast.stmt], param_name: str
    ) -> EffectSet:
        """Collect effects related to uses of a specific parameter."""
        eff = self._empty_effects()
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == param_name:
                    eff = eff.union(EffectSet(
                        effects=frozenset({Effect.READ_HEAP}),
                        read_locations=frozenset({f"{param_name}.{node.attr}"}),
                    ))
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id == param_name:
                    eff = eff.union(_READ_HEAP)
            # Writes: check Assign / AugAssign targets
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    pe = self._param_write_effects(target, param_name)
                    eff = eff.union(pe)
            elif isinstance(node, ast.AugAssign):
                pe = self._param_write_effects(node.target, param_name)
                eff = eff.union(pe)
        return eff

    def _param_write_effects(self, target: ast.expr, param_name: str) -> EffectSet:
        """Check if *target* writes through *param_name*."""
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == param_name:
                return EffectSet(
                    effects=frozenset({Effect.WRITE_HEAP}),
                    write_locations=frozenset({f"{param_name}.{target.attr}"}),
                )
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            if target.value.id == param_name:
                return _WRITE_HEAP
        return _PURE

    # -- return effect collection ------------------------------------

    def _collect_return_effects(self, body: list[ast.stmt]) -> EffectSet:
        """Collect effects from all ``return`` statements."""
        eff = self._empty_effects()
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Return) and node.value:
                eff = eff.union(self.analyze_expression(node.value))
        return eff

    # -- method effect lookup ----------------------------------------

    def _method_effects(self, type_name: str, method_name: str) -> Optional[EffectSet]:
        """Look up effects for well-known method calls on standard types."""
        # Mutating list methods
        list_mutators = {"append", "extend", "insert", "remove", "pop", "clear",
                         "sort", "reverse"}
        list_readers = {"index", "count", "copy"}
        dict_mutators = {"update", "pop", "popitem", "clear", "setdefault"}
        dict_readers = {"get", "keys", "values", "items", "copy"}
        set_mutators = {"add", "discard", "remove", "pop", "clear",
                        "update", "intersection_update", "difference_update",
                        "symmetric_difference_update"}
        set_readers = {"issubset", "issuperset", "isdisjoint", "copy",
                       "union", "intersection", "difference", "symmetric_difference"}
        str_methods = {"upper", "lower", "strip", "lstrip", "rstrip", "split",
                       "join", "replace", "find", "rfind", "index", "rindex",
                       "startswith", "endswith", "encode", "decode", "format",
                       "count", "center", "ljust", "rjust", "zfill", "title",
                       "capitalize", "swapcase", "isalpha", "isdigit", "isalnum",
                       "isspace", "isupper", "islower", "expandtabs", "partition",
                       "rpartition", "maketrans", "translate", "removeprefix",
                       "removesuffix"}

        if type_name in ("list", "[]") and method_name in list_mutators:
            return EffectSet(effects=frozenset({Effect.WRITE_HEAP}))
        if type_name in ("list", "[]") and method_name in list_readers:
            return EffectSet(effects=frozenset({Effect.READ_HEAP}))
        if type_name in ("dict", "{}") and method_name in dict_mutators:
            return EffectSet(effects=frozenset({Effect.WRITE_HEAP}))
        if type_name in ("dict", "{}") and method_name in dict_readers:
            return EffectSet(effects=frozenset({Effect.READ_HEAP}))
        if type_name == "set" and method_name in set_mutators:
            return EffectSet(effects=frozenset({Effect.WRITE_HEAP}))
        if type_name == "set" and method_name in set_readers:
            return EffectSet(effects=frozenset({Effect.READ_HEAP, Effect.ALLOCATE}))
        if type_name == "str" and method_name in str_methods:
            return EffectSet(effects=frozenset({Effect.PURE, Effect.ALLOCATE}))

        return None

    # -- AST utilities -----------------------------------------------

    def _location_of_attribute(self, node: ast.Attribute) -> str:
        """Build a symbolic heap location for an attribute access."""
        base = self._expr_to_str(node.value)
        if base:
            return f"{base}.{node.attr}"
        return f"<expr>.{node.attr}"

    def _expr_to_str(self, node: ast.expr) -> Optional[str]:
        """Best-effort conversion of an expression to a dotted name."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_to_str(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def _exception_name(self, node: ast.expr) -> str:
        """Extract an exception class name from a raise/except node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_to_str(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Call):
            return self._exception_name(node.func)
        return "<unknown>"
