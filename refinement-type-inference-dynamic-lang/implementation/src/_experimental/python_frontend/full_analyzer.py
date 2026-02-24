"""
Full Python AST Analyzer for Refinement Type Inference.

Handles ALL Python constructs with refinement type propagation:
- List/dict/set comprehensions
- Generator expressions
- Class definitions with method refinement types
- Decorators (property, staticmethod, classmethod)
- Context managers (with statements)
- Exception handling (try/except/finally)
- Async/await
- Nested functions with closures
- Multiple return values
- Star args, kwargs
- Walrus operator (:=)
- Match/case statements (Python 3.10+)
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
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


# ===========================================================================
# Local type stubs and data structures
# ===========================================================================

class RefinementKind(Enum):
    """Kind of refinement predicate attached to a type."""
    EXACT = auto()
    RANGE = auto()
    NOTNONE = auto()
    ISINSTANCE = auto()
    LENGTH = auto()
    TRUTHINESS = auto()
    PATTERN = auto()
    MEMBER = auto()
    CALLABLE = auto()
    AWAITABLE = auto()
    CONTEXTMANAGER = auto()
    EXCEPTION = auto()
    LITERAL = auto()
    UNION = auto()
    INTERSECTION = auto()
    NEGATION = auto()


class ScopeKind(Enum):
    """Scope categories within the analyzer."""
    MODULE = auto()
    FUNCTION = auto()
    CLASS = auto()
    COMPREHENSION = auto()
    LAMBDA = auto()
    EXCEPTION_HANDLER = auto()
    WITH_BLOCK = auto()
    MATCH_CASE = auto()


class DecoratorKind(Enum):
    """Recognized decorator types."""
    PROPERTY = auto()
    STATICMETHOD = auto()
    CLASSMETHOD = auto()
    ABSTRACTMETHOD = auto()
    OVERRIDE = auto()
    DATACLASS = auto()
    CUSTOM = auto()


class PatternKind(Enum):
    """Kinds of match-case patterns (3.10+)."""
    CAPTURE = auto()
    LITERAL = auto()
    SEQUENCE = auto()
    MAPPING = auto()
    CLASS = auto()
    STAR = auto()
    OR = auto()
    AS = auto()
    WILDCARD = auto()
    VALUE = auto()
    GUARD = auto()


@dataclass(frozen=True)
class Location:
    """Source location for diagnostics."""
    file: str
    line: int
    col: int
    end_line: Optional[int] = None
    end_col: Optional[int] = None

    @staticmethod
    def from_node(node: ast.AST, file: str = "<unknown>") -> Location:
        return Location(
            file=file,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
            end_col=getattr(node, "end_col_offset", None),
        )


@dataclass
class RefinementType:
    """A base type with zero or more refinement predicates."""
    base: str
    refinements: List[RefinementPredicate] = field(default_factory=list)
    location: Optional[Location] = None

    def add_refinement(self, pred: RefinementPredicate) -> RefinementType:
        """Return a copy with an additional refinement."""
        return RefinementType(
            base=self.base,
            refinements=self.refinements + [pred],
            location=self.location,
        )

    def narrow(self, kind: RefinementKind, value: Any) -> RefinementType:
        """Convenience: add a predicate of the given kind."""
        return self.add_refinement(RefinementPredicate(kind=kind, value=value))

    def join(self, other: RefinementType) -> RefinementType:
        """Upper bound (union) of two refinement types."""
        if self.base == other.base:
            shared = [r for r in self.refinements if r in other.refinements]
            return RefinementType(base=self.base, refinements=shared)
        return RefinementType(base=f"Union[{self.base}, {other.base}]")

    def meet(self, other: RefinementType) -> RefinementType:
        """Lower bound (intersection) of two refinement types."""
        if self.base == other.base:
            combined = list(set(self.refinements) | set(other.refinements))
            return RefinementType(base=self.base, refinements=combined)
        return RefinementType(base="Never")


@dataclass(frozen=True)
class RefinementPredicate:
    """Single refinement predicate on a type."""
    kind: RefinementKind
    value: Any = None
    negated: bool = False

    def negate(self) -> RefinementPredicate:
        return RefinementPredicate(
            kind=self.kind, value=self.value, negated=not self.negated
        )


@dataclass
class TypeEnv:
    """Typing environment with scoped variable bindings."""
    bindings: Dict[str, RefinementType] = field(default_factory=dict)
    parent: Optional[TypeEnv] = None
    scope_kind: ScopeKind = ScopeKind.MODULE

    def lookup(self, name: str) -> Optional[RefinementType]:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def bind(self, name: str, typ: RefinementType) -> None:
        self.bindings[name] = typ

    def child(self, scope_kind: ScopeKind) -> TypeEnv:
        return TypeEnv(parent=self, scope_kind=scope_kind)

    def snapshot(self) -> Dict[str, RefinementType]:
        result: Dict[str, RefinementType] = {}
        if self.parent is not None:
            result.update(self.parent.snapshot())
        result.update(self.bindings)
        return result


@dataclass
class AnalysisResult:
    """Result of analyzing an AST node or subtree."""
    typ: RefinementType
    env: TypeEnv
    effects: List[str] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    reachable: bool = True


@dataclass
class Diagnostic:
    """An analysis diagnostic (warning, error, info)."""
    message: str
    location: Location
    severity: str = "warning"


UNKNOWN_TYPE = RefinementType(base="Unknown")
NONE_TYPE = RefinementType(base="None")
BOOL_TYPE = RefinementType(base="bool")
INT_TYPE = RefinementType(base="int")
FLOAT_TYPE = RefinementType(base="float")
STR_TYPE = RefinementType(base="str")
BYTES_TYPE = RefinementType(base="bytes")
LIST_TYPE = RefinementType(base="list")
DICT_TYPE = RefinementType(base="dict")
SET_TYPE = RefinementType(base="set")
TUPLE_TYPE = RefinementType(base="tuple")
NEVER_TYPE = RefinementType(base="Never")

_LITERAL_MAP: Dict[type, RefinementType] = {
    int: INT_TYPE,
    float: FLOAT_TYPE,
    str: STR_TYPE,
    bytes: BYTES_TYPE,
    bool: BOOL_TYPE,
    type(None): NONE_TYPE,
}


# ===========================================================================
# ComprehensionAnalyzer
# ===========================================================================

class ComprehensionAnalyzer:
    """Analyze list/dict/set comprehensions and generator expressions.

    Propagates refinements through comprehension filters (if-clauses)
    and tracks bound variables introduced by the comprehension target.
    """

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._bound_vars: Dict[str, RefinementType] = {}
        self._diagnostics: List[Diagnostic] = []

    def analyze_listcomp(
        self, node: ast.ListComp, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a list comprehension: [expr for ... in ... if ...]."""
        inner_env = env.child(ScopeKind.COMPREHENSION)
        for gen in node.generators:
            inner_env = self._analyze_generator(gen, inner_env)
        elt_result = self._analyze_element(node.elt, inner_env)
        result_type = RefinementType(
            base=f"list[{elt_result.typ.base}]",
            location=Location.from_node(node, self._file),
        )
        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    def analyze_setcomp(
        self, node: ast.SetComp, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a set comprehension: {expr for ... in ... if ...}."""
        inner_env = env.child(ScopeKind.COMPREHENSION)
        for gen in node.generators:
            inner_env = self._analyze_generator(gen, inner_env)
        elt_result = self._analyze_element(node.elt, inner_env)
        result_type = RefinementType(
            base=f"set[{elt_result.typ.base}]",
            location=Location.from_node(node, self._file),
        )
        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    def analyze_dictcomp(
        self, node: ast.DictComp, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a dict comprehension: {k: v for ... in ... if ...}."""
        inner_env = env.child(ScopeKind.COMPREHENSION)
        for gen in node.generators:
            inner_env = self._analyze_generator(gen, inner_env)
        key_result = self._analyze_element(node.key, inner_env)
        val_result = self._analyze_element(node.value, inner_env)
        result_type = RefinementType(
            base=f"dict[{key_result.typ.base}, {val_result.typ.base}]",
            location=Location.from_node(node, self._file),
        )
        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    def analyze_generatorexp(
        self, node: ast.GeneratorExp, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a generator expression: (expr for ... in ... if ...)."""
        inner_env = env.child(ScopeKind.COMPREHENSION)
        for gen in node.generators:
            inner_env = self._analyze_generator(gen, inner_env)
        elt_result = self._analyze_element(node.elt, inner_env)
        result_type = RefinementType(
            base=f"Generator[{elt_result.typ.base}]",
            location=Location.from_node(node, self._file),
        )
        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    # -- internal helpers ---------------------------------------------------

    def _analyze_generator(
        self, gen: ast.comprehension, env: TypeEnv
    ) -> TypeEnv:
        """Process a single generator clause (for ... in ... if ...)."""
        iter_type = self._infer_iter_element_type(gen.iter, env)
        env = self._bind_target(gen.target, iter_type, env)
        for if_clause in gen.ifs:
            env = self._apply_guard(if_clause, env)
        if gen.is_async:
            iter_type = RefinementType(
                base=f"AsyncIterElement[{iter_type.base}]"
            )
        return env

    def _bind_target(
        self, target: ast.AST, typ: RefinementType, env: TypeEnv
    ) -> TypeEnv:
        """Bind the comprehension target variable(s) in *env*."""
        if isinstance(target, ast.Name):
            env.bind(target.id, typ)
            self._bound_vars[target.id] = typ
        elif isinstance(target, ast.Tuple):
            for i, elt in enumerate(target.elts):
                inner_type = RefinementType(base=f"Element_{i}[{typ.base}]")
                env = self._bind_target(elt, inner_type, env)
        elif isinstance(target, ast.Starred):
            rest_type = RefinementType(base=f"list[{typ.base}]")
            env = self._bind_target(target.value, rest_type, env)
        elif isinstance(target, ast.List):
            for i, elt in enumerate(target.elts):
                inner_type = RefinementType(base=f"Element_{i}[{typ.base}]")
                env = self._bind_target(elt, inner_type, env)
        return env

    def _infer_iter_element_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Infer the element type of an iterable expression."""
        if isinstance(node, ast.Name):
            container = env.lookup(node.id)
            if container is not None:
                base = container.base
                if base.startswith("list["):
                    return RefinementType(base=base[5:-1])
                if base.startswith("set["):
                    return RefinementType(base=base[4:-1])
                if base.startswith("dict["):
                    key_part = base[5:].split(",")[0].strip()
                    return RefinementType(base=key_part)
                return RefinementType(base=f"ElementOf[{base}]")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "range":
                return INT_TYPE
            if isinstance(node.func, ast.Name) and node.func.id == "enumerate":
                return RefinementType(base="tuple[int, Unknown]")
            if isinstance(node.func, ast.Name) and node.func.id == "zip":
                return TUPLE_TYPE
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return STR_TYPE
            if isinstance(node.value, bytes):
                return BYTES_TYPE
        return UNKNOWN_TYPE

    def _apply_guard(self, node: ast.AST, env: TypeEnv) -> TypeEnv:
        """Narrow refinements from an if-clause in a comprehension."""
        if isinstance(node, ast.Compare):
            return self._apply_comparison_guard(node, env)
        if isinstance(node, ast.Call):
            return self._apply_call_guard(node, env)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            negated_env = self._apply_guard(node.operand, env)
            return negated_env
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for val in node.values:
                    env = self._apply_guard(val, env)
                return env
        if isinstance(node, ast.Name):
            existing = env.lookup(node.id)
            if existing is not None:
                env.bind(
                    node.id,
                    existing.narrow(RefinementKind.TRUTHINESS, True),
                )
        return env

    def _apply_comparison_guard(
        self, node: ast.Compare, env: TypeEnv
    ) -> TypeEnv:
        """Apply refinement from a comparison expression."""
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(left, ast.Name):
                existing = env.lookup(left.id) or UNKNOWN_TYPE
                if isinstance(op, ast.IsNot) and isinstance(
                    comparator, ast.Constant
                ) and comparator.value is None:
                    env.bind(
                        left.id,
                        existing.narrow(RefinementKind.NOTNONE, True),
                    )
                elif isinstance(op, (ast.Gt, ast.GtE)):
                    env.bind(
                        left.id,
                        existing.narrow(RefinementKind.RANGE, (">=", comparator)),
                    )
                elif isinstance(op, (ast.Lt, ast.LtE)):
                    env.bind(
                        left.id,
                        existing.narrow(RefinementKind.RANGE, ("<=", comparator)),
                    )
                elif isinstance(op, ast.In):
                    env.bind(
                        left.id,
                        existing.narrow(RefinementKind.MEMBER, comparator),
                    )
            left = comparator
        return env

    def _apply_call_guard(self, node: ast.Call, env: TypeEnv) -> TypeEnv:
        """Apply refinement from a call used as a guard (e.g. isinstance)."""
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
        ):
            target = node.args[0]
            type_arg = node.args[1]
            if isinstance(target, ast.Name):
                type_name = self._extract_type_name(type_arg)
                existing = env.lookup(target.id) or UNKNOWN_TYPE
                env.bind(
                    target.id,
                    existing.narrow(RefinementKind.ISINSTANCE, type_name),
                )
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "callable"
            and len(node.args) == 1
        ):
            target = node.args[0]
            if isinstance(target, ast.Name):
                existing = env.lookup(target.id) or UNKNOWN_TYPE
                env.bind(
                    target.id,
                    existing.narrow(RefinementKind.CALLABLE, True),
                )
        return env

    def _extract_type_name(self, node: ast.AST) -> str:
        """Extract a type name from an isinstance second argument."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._extract_type_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Tuple):
            parts = [self._extract_type_name(e) for e in node.elts]
            return f"Union[{', '.join(parts)}]"
        return "Unknown"

    @property
    def bound_variables(self) -> Dict[str, RefinementType]:
        """Variables bound during the most recent analysis."""
        return dict(self._bound_vars)


# ===========================================================================
# ClassAnalyzer
# ===========================================================================

class ClassAnalyzer:
    """Analyze class definitions including method signatures, properties,
    inheritance, MRO computation, and descriptor protocol handling."""

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._class_registry: Dict[str, ClassInfo] = {}
        self._diagnostics: List[Diagnostic] = []

    def analyze_classdef(
        self, node: ast.ClassDef, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a full class definition."""
        loc = Location.from_node(node, self._file)
        bases = [self._resolve_base(b, env) for b in node.bases]
        mro = self._compute_mro(node.name, bases)

        info = ClassInfo(
            name=node.name,
            location=loc,
            bases=bases,
            mro=mro,
        )

        for decorator in node.decorator_list:
            info.decorators.append(self._classify_decorator(decorator))

        class_env = env.child(ScopeKind.CLASS)
        class_env.bind("__class__", RefinementType(base=f"type[{node.name}]"))

        for stmt in node.body:
            self._analyze_class_body_stmt(stmt, info, class_env)

        self._check_abstract_methods(info)
        self._class_registry[node.name] = info

        class_type = RefinementType(base=f"type[{node.name}]", location=loc)
        env.bind(node.name, class_type)
        return AnalysisResult(
            typ=class_type, env=env, diagnostics=self._diagnostics[:]
        )

    def _analyze_class_body_stmt(
        self, stmt: ast.AST, info: ClassInfo, env: TypeEnv
    ) -> None:
        """Process a single statement inside a class body."""
        if isinstance(stmt, ast.FunctionDef) or isinstance(
            stmt, ast.AsyncFunctionDef
        ):
            self._analyze_method(stmt, info, env)
        elif isinstance(stmt, ast.AnnAssign):
            self._analyze_class_annotation(stmt, info, env)
        elif isinstance(stmt, ast.Assign):
            self._analyze_class_assign(stmt, info, env)
        elif isinstance(stmt, ast.ClassDef):
            nested_analyzer = ClassAnalyzer(self._file)
            nested_analyzer.analyze_classdef(stmt, env)
            self._class_registry.update(nested_analyzer._class_registry)
        elif isinstance(stmt, ast.Pass):
            pass
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            pass  # docstring
        else:
            self._diagnostics.append(Diagnostic(
                message=f"Unhandled class body statement: {type(stmt).__name__}",
                location=Location.from_node(stmt, self._file),
                severity="info",
            ))

    def _analyze_method(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        info: ClassInfo,
        env: TypeEnv,
    ) -> None:
        """Analyze a method definition within a class."""
        decorators = [self._classify_decorator(d) for d in node.decorator_list]
        is_static = DecoratorKind.STATICMETHOD in decorators
        is_classmethod = DecoratorKind.CLASSMETHOD in decorators
        is_property = DecoratorKind.PROPERTY in decorators
        is_async = isinstance(node, ast.AsyncFunctionDef)

        param_types = self._extract_param_types(node.args, info.name, is_static)
        return_type = self._extract_return_type(node)

        method_info = MethodInfo(
            name=node.name,
            params=param_types,
            return_type=return_type,
            decorators=decorators,
            is_static=is_static,
            is_classmethod=is_classmethod,
            is_property=is_property,
            is_async=is_async,
            location=Location.from_node(node, self._file),
        )

        if is_property:
            info.properties[node.name] = return_type
        info.methods[node.name] = method_info

        if node.name == "__init__":
            self._analyze_init_body(node.body, info, env)

    def _analyze_init_body(
        self, body: List[ast.stmt], info: ClassInfo, env: TypeEnv
    ) -> None:
        """Extract instance variable assignments from __init__."""
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        info.instance_vars[target.attr] = UNKNOWN_TYPE
            elif isinstance(stmt, ast.AnnAssign):
                if (
                    isinstance(stmt.target, ast.Attribute)
                    and isinstance(stmt.target.value, ast.Name)
                    and stmt.target.value.id == "self"
                ):
                    ann_type = self._annotation_to_type(stmt.annotation)
                    info.instance_vars[stmt.target.attr] = ann_type
            elif isinstance(stmt, ast.If):
                self._analyze_init_body(stmt.body, info, env)
                self._analyze_init_body(stmt.orelse, info, env)
            elif isinstance(stmt, ast.For):
                self._analyze_init_body(stmt.body, info, env)

    def _analyze_class_annotation(
        self, stmt: ast.AnnAssign, info: ClassInfo, env: TypeEnv
    ) -> None:
        """Handle annotated assignments at class level."""
        if isinstance(stmt.target, ast.Name):
            ann_type = self._annotation_to_type(stmt.annotation)
            info.class_vars[stmt.target.id] = ann_type
            env.bind(stmt.target.id, ann_type)

    def _analyze_class_assign(
        self, stmt: ast.Assign, info: ClassInfo, env: TypeEnv
    ) -> None:
        """Handle plain assignments at class level."""
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                info.class_vars[target.id] = UNKNOWN_TYPE
                env.bind(target.id, UNKNOWN_TYPE)

    def _extract_param_types(
        self, args: ast.arguments, class_name: str, is_static: bool
    ) -> List[Tuple[str, RefinementType]]:
        """Extract parameter names and types from a method signature."""
        params: List[Tuple[str, RefinementType]] = []
        all_args = args.posonlyargs + args.args
        for i, arg in enumerate(all_args):
            if i == 0 and not is_static:
                params.append((arg.arg, RefinementType(base=class_name)))
            elif arg.annotation:
                params.append((arg.arg, self._annotation_to_type(arg.annotation)))
            else:
                params.append((arg.arg, UNKNOWN_TYPE))
        if args.vararg:
            params.append((f"*{args.vararg.arg}", RefinementType(base="tuple[Unknown, ...]")))
        for arg in args.kwonlyargs:
            ann_type = self._annotation_to_type(arg.annotation) if arg.annotation else UNKNOWN_TYPE
            params.append((arg.arg, ann_type))
        if args.kwarg:
            params.append((f"**{args.kwarg.arg}", RefinementType(base="dict[str, Unknown]")))
        return params

    def _extract_return_type(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> RefinementType:
        """Extract the return type annotation of a function/method."""
        if node.returns:
            return self._annotation_to_type(node.returns)
        return UNKNOWN_TYPE

    def _annotation_to_type(self, node: ast.AST) -> RefinementType:
        """Convert an annotation AST node to a RefinementType."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return RefinementType(base=node.value)
        if isinstance(node, ast.Name):
            return RefinementType(base=node.id)
        if isinstance(node, ast.Attribute):
            return RefinementType(base=ast.dump(node))
        if isinstance(node, ast.Subscript):
            base_name = self._annotation_to_type(node.value).base
            return RefinementType(base=f"{base_name}[...]")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._annotation_to_type(node.left)
            right = self._annotation_to_type(node.right)
            return RefinementType(base=f"Union[{left.base}, {right.base}]")
        return UNKNOWN_TYPE

    def _classify_decorator(self, node: ast.AST) -> DecoratorKind:
        """Determine the kind of decorator from its AST node."""
        if isinstance(node, ast.Name):
            mapping = {
                "property": DecoratorKind.PROPERTY,
                "staticmethod": DecoratorKind.STATICMETHOD,
                "classmethod": DecoratorKind.CLASSMETHOD,
                "abstractmethod": DecoratorKind.ABSTRACTMETHOD,
                "override": DecoratorKind.OVERRIDE,
                "dataclass": DecoratorKind.DATACLASS,
            }
            return mapping.get(node.id, DecoratorKind.CUSTOM)
        if isinstance(node, ast.Attribute):
            return self._classify_decorator(
                ast.Name(id=node.attr, ctx=ast.Load())
            )
        if isinstance(node, ast.Call):
            return self._classify_decorator(node.func)
        return DecoratorKind.CUSTOM

    def _resolve_base(self, node: ast.AST, env: TypeEnv) -> str:
        """Resolve a base class name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._resolve_base(node.value, env)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{self._resolve_base(node.value, env)}[...]"
        return "Unknown"

    def _compute_mro(self, name: str, bases: List[str]) -> List[str]:
        """Compute a simplified method resolution order (C3 linearization)."""
        mro = [name]
        for base in bases:
            if base in self._class_registry:
                base_mro = self._class_registry[base].mro
                for cls in base_mro:
                    if cls not in mro:
                        mro.append(cls)
            elif base not in mro:
                mro.append(base)
        if "object" not in mro:
            mro.append("object")
        return mro

    def _check_abstract_methods(self, info: ClassInfo) -> None:
        """Warn if concrete class has unimplemented abstract methods."""
        is_abstract = DecoratorKind.ABSTRACTMETHOD in info.decorators
        if is_abstract:
            return
        for base_name in info.mro[1:]:
            base_info = self._class_registry.get(base_name)
            if base_info is None:
                continue
            for mname, minfo in base_info.methods.items():
                if (
                    DecoratorKind.ABSTRACTMETHOD in minfo.decorators
                    and mname not in info.methods
                ):
                    self._diagnostics.append(Diagnostic(
                        message=(
                            f"Class {info.name} does not implement abstract "
                            f"method {mname} from {base_name}"
                        ),
                        location=info.location,
                    ))

    def lookup_class(self, name: str) -> Optional[ClassInfo]:
        """Look up a previously analyzed class by name."""
        return self._class_registry.get(name)


@dataclass
class ClassInfo:
    """Collected information about an analyzed class."""
    name: str
    location: Location
    bases: List[str] = field(default_factory=list)
    mro: List[str] = field(default_factory=list)
    methods: Dict[str, MethodInfo] = field(default_factory=dict)
    properties: Dict[str, RefinementType] = field(default_factory=dict)
    class_vars: Dict[str, RefinementType] = field(default_factory=dict)
    instance_vars: Dict[str, RefinementType] = field(default_factory=dict)
    decorators: List[DecoratorKind] = field(default_factory=list)


@dataclass
class MethodInfo:
    """Collected information about a method."""
    name: str
    params: List[Tuple[str, RefinementType]]
    return_type: RefinementType
    decorators: List[DecoratorKind]
    is_static: bool
    is_classmethod: bool
    is_property: bool
    is_async: bool
    location: Location


# ===========================================================================
# ContextManagerAnalyzer
# ===========================================================================

class ContextManagerAnalyzer:
    """Analyze with-statements: __enter__/__exit__ protocol,
    resource tracking, exception propagation, nested with blocks."""

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._active_resources: Deque[ResourceInfo] = deque()
        self._diagnostics: List[Diagnostic] = []

    def analyze_with(
        self, node: ast.With, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a with-statement and its body."""
        loc = Location.from_node(node, self._file)
        body_env = env.child(ScopeKind.WITH_BLOCK)

        for item in node.items:
            body_env = self._analyze_with_item(item, body_env, is_async=False)

        body_results = self._analyze_body(node.body, body_env)

        for _ in node.items:
            self._pop_resource()

        return AnalysisResult(
            typ=NONE_TYPE,
            env=env,
            effects=[f"with_block@{loc.line}"],
            diagnostics=self._diagnostics[:],
            reachable=body_results.reachable,
        )

    def analyze_async_with(
        self, node: ast.AsyncWith, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an async with-statement."""
        loc = Location.from_node(node, self._file)
        body_env = env.child(ScopeKind.WITH_BLOCK)

        for item in node.items:
            body_env = self._analyze_with_item(item, body_env, is_async=True)

        body_results = self._analyze_body(node.body, body_env)

        for _ in node.items:
            self._pop_resource()

        return AnalysisResult(
            typ=NONE_TYPE,
            env=env,
            effects=[f"async_with_block@{loc.line}"],
            diagnostics=self._diagnostics[:],
            reachable=body_results.reachable,
        )

    def _analyze_with_item(
        self, item: ast.withitem, env: TypeEnv, *, is_async: bool
    ) -> TypeEnv:
        """Analyze a single with-item (context_expr as optional_var)."""
        cm_type = self._infer_context_manager_type(item.context_expr, env)
        enter_type = self._resolve_enter_type(cm_type, is_async=is_async)

        resource = ResourceInfo(
            cm_type=cm_type,
            enter_type=enter_type,
            var_name=None,
            is_async=is_async,
            location=Location.from_node(item.context_expr, self._file),
        )

        if item.optional_vars is not None:
            if isinstance(item.optional_vars, ast.Name):
                resource.var_name = item.optional_vars.id
                env.bind(item.optional_vars.id, enter_type)
            elif isinstance(item.optional_vars, ast.Tuple):
                for i, elt in enumerate(item.optional_vars.elts):
                    if isinstance(elt, ast.Name):
                        elem_type = RefinementType(
                            base=f"Element_{i}[{enter_type.base}]"
                        )
                        env.bind(elt.id, elem_type)

        self._active_resources.append(resource)
        return env

    def _infer_context_manager_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Infer the type of a context manager expression."""
        if isinstance(node, ast.Name):
            looked = env.lookup(node.id)
            if looked is not None:
                return looked.narrow(RefinementKind.CONTEXTMANAGER, True)
            return RefinementType(base=node.id).narrow(
                RefinementKind.CONTEXTMANAGER, True
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return RefinementType(base=node.func.id).narrow(
                    RefinementKind.CONTEXTMANAGER, True
                )
            if isinstance(node.func, ast.Attribute):
                return RefinementType(base=node.func.attr).narrow(
                    RefinementKind.CONTEXTMANAGER, True
                )
        return UNKNOWN_TYPE.narrow(RefinementKind.CONTEXTMANAGER, True)

    def _resolve_enter_type(
        self, cm_type: RefinementType, *, is_async: bool
    ) -> RefinementType:
        """Determine the type yielded by __enter__ / __aenter__."""
        base = cm_type.base
        known_mappings: Dict[str, str] = {
            "open": "TextIOWrapper",
            "Lock": "bool",
            "RLock": "bool",
            "Semaphore": "bool",
            "Condition": "bool",
            "suppress": "None",
            "closing": base,
            "nullcontext": "None",
        }
        resolved = known_mappings.get(base)
        if resolved is not None:
            result = RefinementType(base=resolved)
        else:
            result = RefinementType(base=f"EnterType[{base}]")
        if is_async:
            result = RefinementType(base=f"Awaited[{result.base}]")
        return result

    def _analyze_body(
        self, body: List[ast.stmt], env: TypeEnv
    ) -> AnalysisResult:
        """Analyze the body of a with-block."""
        reachable = True
        for stmt in body:
            if isinstance(stmt, ast.Return):
                reachable = False
            elif isinstance(stmt, ast.Raise):
                reachable = False
            elif isinstance(stmt, ast.Break):
                reachable = False
            elif isinstance(stmt, ast.Continue):
                reachable = False
        return AnalysisResult(typ=NONE_TYPE, env=env, reachable=reachable)

    def _pop_resource(self) -> Optional[ResourceInfo]:
        """Pop the most recently entered resource."""
        if self._active_resources:
            return self._active_resources.pop()
        return None

    @property
    def active_resources(self) -> List[ResourceInfo]:
        """Currently active context-managed resources."""
        return list(self._active_resources)


@dataclass
class ResourceInfo:
    """Information about an active context-managed resource."""
    cm_type: RefinementType
    enter_type: RefinementType
    var_name: Optional[str]
    is_async: bool
    location: Location


# ===========================================================================
# ExceptionFlowAnalyzer
# ===========================================================================

class ExceptionFlowAnalyzer:
    """Analyze try/except/finally blocks with exception type propagation,
    handler matching, exception variable refinement, and chaining."""

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._diagnostics: List[Diagnostic] = []
        self._exception_stack: Deque[RefinementType] = deque()

    def analyze_try(
        self, node: ast.Try, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a try/except/else/finally statement."""
        loc = Location.from_node(node, self._file)

        try_env = env.child(ScopeKind.EXCEPTION_HANDLER)
        try_result = self._analyze_block(node.body, try_env)
        raised_types = self._collect_raised_types(node.body)

        handler_results: List[AnalysisResult] = []
        handler_envs: List[TypeEnv] = []
        has_bare_except = False

        for handler in node.handlers:
            h_result, h_env = self._analyze_handler(
                handler, env, raised_types
            )
            handler_results.append(h_result)
            handler_envs.append(h_env)
            if handler.type is None:
                has_bare_except = True

        if has_bare_except and len(node.handlers) > 1:
            last = node.handlers[-1]
            if last.type is not None:
                self._diagnostics.append(Diagnostic(
                    message="Bare except should be last handler",
                    location=Location.from_node(last, self._file),
                ))

        else_result: Optional[AnalysisResult] = None
        if node.orelse:
            else_result = self._analyze_block(node.orelse, try_env)

        finally_result: Optional[AnalysisResult] = None
        if node.finalbody:
            finally_result = self._analyze_block(node.finalbody, env)

        merged_env = self._merge_exception_envs(env, handler_envs, try_env)
        all_reachable = try_result.reachable or any(
            r.reachable for r in handler_results
        )

        return AnalysisResult(
            typ=NONE_TYPE,
            env=merged_env,
            effects=[f"try_block@{loc.line}"],
            diagnostics=self._diagnostics[:],
            reachable=all_reachable,
        )

    def analyze_try_star(
        self, node: ast.AST, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a try/except* statement (Python 3.11+)."""
        loc = Location.from_node(node, self._file)
        body = getattr(node, "body", [])
        handlers = getattr(node, "handlers", [])

        try_env = env.child(ScopeKind.EXCEPTION_HANDLER)
        try_result = self._analyze_block(body, try_env)
        raised_types = self._collect_raised_types(body)

        handler_envs: List[TypeEnv] = []
        for handler in handlers:
            h_result, h_env = self._analyze_handler(handler, env, raised_types)
            handler_envs.append(h_env)

        merged_env = self._merge_exception_envs(env, handler_envs, try_env)
        return AnalysisResult(
            typ=NONE_TYPE,
            env=merged_env,
            effects=[f"try_star@{loc.line}"],
            diagnostics=self._diagnostics[:],
        )

    def analyze_raise(
        self, node: ast.Raise, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a raise statement."""
        loc = Location.from_node(node, self._file)
        if node.exc is not None:
            exc_type = self._infer_exception_type(node.exc, env)
            self._exception_stack.append(exc_type)
            if node.cause is not None:
                cause_type = self._infer_exception_type(node.cause, env)
                exc_type = exc_type.narrow(
                    RefinementKind.EXCEPTION, f"chained_from:{cause_type.base}"
                )
        else:
            if not self._exception_stack:
                self._diagnostics.append(Diagnostic(
                    message="Bare raise outside of exception handler",
                    location=loc,
                    severity="error",
                ))

        return AnalysisResult(
            typ=NEVER_TYPE, env=env, reachable=False,
            diagnostics=self._diagnostics[:],
        )

    def _analyze_handler(
        self,
        handler: ast.ExceptHandler,
        env: TypeEnv,
        raised_types: List[RefinementType],
    ) -> Tuple[AnalysisResult, TypeEnv]:
        """Analyze a single except handler clause."""
        handler_env = env.child(ScopeKind.EXCEPTION_HANDLER)

        if handler.type is not None:
            exc_type = self._resolve_exception_type(handler.type)
            if handler.name is not None:
                handler_env.bind(handler.name, exc_type)
        else:
            exc_type = RefinementType(base="BaseException")
            if handler.name is not None:
                handler_env.bind(handler.name, exc_type)

        result = self._analyze_block(handler.body, handler_env)

        if handler.name is not None:
            handler_env.bind(handler.name, NONE_TYPE)

        return result, handler_env

    def _collect_raised_types(
        self, body: List[ast.stmt]
    ) -> List[RefinementType]:
        """Collect exception types raised in a block of statements."""
        raised: List[RefinementType] = []
        for stmt in body:
            if isinstance(stmt, ast.Raise) and stmt.exc is not None:
                raised.append(self._infer_exception_type(stmt.exc, TypeEnv()))
            for child in ast.walk(stmt):
                if isinstance(child, ast.Raise) and child.exc is not None:
                    raised.append(
                        self._infer_exception_type(child.exc, TypeEnv())
                    )
        return raised

    def _infer_exception_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Infer the type of a raised exception expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return RefinementType(base=node.func.id).narrow(
                    RefinementKind.EXCEPTION, True
                )
            if isinstance(node.func, ast.Attribute):
                return RefinementType(base=node.func.attr).narrow(
                    RefinementKind.EXCEPTION, True
                )
        if isinstance(node, ast.Name):
            looked = env.lookup(node.id)
            if looked is not None:
                return looked
            return RefinementType(base=node.id).narrow(
                RefinementKind.EXCEPTION, True
            )
        return RefinementType(base="BaseException").narrow(
            RefinementKind.EXCEPTION, True
        )

    def _resolve_exception_type(self, node: ast.AST) -> RefinementType:
        """Resolve exception type from an except clause type annotation."""
        if isinstance(node, ast.Name):
            return RefinementType(base=node.id).narrow(
                RefinementKind.EXCEPTION, True
            )
        if isinstance(node, ast.Attribute):
            return RefinementType(
                base=f"{ast.dump(node)}"
            ).narrow(RefinementKind.EXCEPTION, True)
        if isinstance(node, ast.Tuple):
            parts = []
            for elt in node.elts:
                t = self._resolve_exception_type(elt)
                parts.append(t.base)
            return RefinementType(
                base=f"Union[{', '.join(parts)}]"
            ).narrow(RefinementKind.EXCEPTION, True)
        return RefinementType(base="BaseException").narrow(
            RefinementKind.EXCEPTION, True
        )

    def _analyze_block(
        self, stmts: List[ast.stmt], env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a block of statements for exception flow."""
        reachable = True
        for stmt in stmts:
            if isinstance(stmt, ast.Raise):
                reachable = False
                break
            if isinstance(stmt, ast.Return):
                reachable = False
                break
        return AnalysisResult(typ=NONE_TYPE, env=env, reachable=reachable)

    def _merge_exception_envs(
        self,
        outer: TypeEnv,
        handler_envs: List[TypeEnv],
        try_env: TypeEnv,
    ) -> TypeEnv:
        """Merge environments from try body and exception handlers."""
        merged = outer.child(outer.scope_kind)
        all_envs = [try_env] + handler_envs
        all_names: Set[str] = set()
        for e in all_envs:
            all_names.update(e.bindings.keys())

        for name in all_names:
            types = []
            for e in all_envs:
                t = e.bindings.get(name)
                if t is not None:
                    types.append(t)
            if len(types) == 1:
                merged.bind(name, types[0])
            elif len(types) > 1:
                joined = types[0]
                for t in types[1:]:
                    joined = joined.join(t)
                merged.bind(name, joined)
        return merged


# ===========================================================================
# AsyncAnalyzer
# ===========================================================================

class AsyncAnalyzer:
    """Analyze async/await constructs: coroutine types, async for,
    async with, async generators, and awaitable refinement."""

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._diagnostics: List[Diagnostic] = []
        self._in_async_context: bool = False
        self._coroutine_depth: int = 0

    def analyze_async_funcdef(
        self, node: ast.AsyncFunctionDef, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an async function definition."""
        loc = Location.from_node(node, self._file)
        prev_async = self._in_async_context
        self._in_async_context = True
        self._coroutine_depth += 1

        func_env = env.child(ScopeKind.FUNCTION)
        self._bind_params(node.args, func_env)

        has_yield = self._contains_yield(node.body)
        return_type = self._extract_return_annotation(node)

        if has_yield:
            result_type = RefinementType(
                base=f"AsyncGenerator[{return_type.base}, None]",
                location=loc,
            )
        else:
            result_type = RefinementType(
                base=f"Coroutine[Any, Any, {return_type.base}]",
                location=loc,
            )

        result_type = result_type.narrow(RefinementKind.AWAITABLE, True)
        env.bind(node.name, result_type)

        self._in_async_context = prev_async
        self._coroutine_depth -= 1

        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    def analyze_await(
        self, node: ast.Await, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an await expression."""
        loc = Location.from_node(node, self._file)

        if not self._in_async_context:
            self._diagnostics.append(Diagnostic(
                message="'await' used outside async function",
                location=loc,
                severity="error",
            ))

        awaited_type = self._infer_awaited_type(node.value, env)
        return AnalysisResult(
            typ=awaited_type, env=env, diagnostics=self._diagnostics[:]
        )

    def analyze_async_for(
        self, node: ast.AsyncFor, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an async for loop."""
        loc = Location.from_node(node, self._file)

        if not self._in_async_context:
            self._diagnostics.append(Diagnostic(
                message="'async for' used outside async function",
                location=loc,
                severity="error",
            ))

        iter_type = self._infer_async_iter_type(node.iter, env)
        body_env = env.child(ScopeKind.FUNCTION)
        self._bind_target(node.target, iter_type, body_env)

        return AnalysisResult(
            typ=NONE_TYPE,
            env=env,
            effects=[f"async_for@{loc.line}"],
            diagnostics=self._diagnostics[:],
        )

    def analyze_yield(
        self, node: ast.Yield, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a yield expression."""
        loc = Location.from_node(node, self._file)
        if node.value is not None:
            yield_type = self._infer_expr_type(node.value, env)
        else:
            yield_type = NONE_TYPE
        result_type = RefinementType(base=f"YieldType[{yield_type.base}]")
        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    def analyze_yield_from(
        self, node: ast.YieldFrom, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a yield from expression."""
        sub_type = self._infer_expr_type(node.value, env)
        result_type = RefinementType(base=f"YieldFrom[{sub_type.base}]")
        return AnalysisResult(
            typ=result_type, env=env, diagnostics=self._diagnostics[:]
        )

    # -- internal helpers ---------------------------------------------------

    def _infer_awaited_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Unwrap the awaited type from a coroutine/awaitable."""
        raw = self._infer_expr_type(node, env)
        base = raw.base
        if base.startswith("Coroutine["):
            parts = base[10:-1].split(",")
            if len(parts) >= 3:
                return RefinementType(base=parts[-1].strip())
        if base.startswith("Awaitable["):
            return RefinementType(base=base[10:-1])
        if base.startswith("Task["):
            return RefinementType(base=base[5:-1])
        return RefinementType(base=f"Awaited[{base}]")

    def _infer_async_iter_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Infer the element type from an async iterable."""
        raw = self._infer_expr_type(node, env)
        base = raw.base
        if base.startswith("AsyncGenerator["):
            return RefinementType(base=base[15:].split(",")[0].strip())
        if base.startswith("AsyncIterable["):
            return RefinementType(base=base[14:-1])
        return RefinementType(base=f"AsyncElement[{base}]")

    def _infer_expr_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Basic expression type inference (simplified)."""
        if isinstance(node, ast.Name):
            return env.lookup(node.id) or UNKNOWN_TYPE
        if isinstance(node, ast.Constant):
            return _LITERAL_MAP.get(type(node.value), UNKNOWN_TYPE)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return RefinementType(base=f"ReturnOf[{node.func.id}]")
        if isinstance(node, ast.Attribute):
            return RefinementType(base=f"AttrOf[{node.attr}]")
        return UNKNOWN_TYPE

    def _bind_params(self, args: ast.arguments, env: TypeEnv) -> None:
        """Bind function parameters into the environment."""
        for arg in args.posonlyargs + args.args:
            if arg.annotation:
                env.bind(arg.arg, RefinementType(base=ast.dump(arg.annotation)))
            else:
                env.bind(arg.arg, UNKNOWN_TYPE)
        if args.vararg:
            env.bind(args.vararg.arg, RefinementType(base="tuple[Unknown, ...]"))
        for arg in args.kwonlyargs:
            env.bind(arg.arg, UNKNOWN_TYPE)
        if args.kwarg:
            env.bind(args.kwarg.arg, RefinementType(base="dict[str, Unknown]"))

    def _bind_target(
        self, target: ast.AST, typ: RefinementType, env: TypeEnv
    ) -> None:
        """Bind a for-loop target variable."""
        if isinstance(target, ast.Name):
            env.bind(target.id, typ)
        elif isinstance(target, ast.Tuple):
            for i, elt in enumerate(target.elts):
                self._bind_target(
                    elt,
                    RefinementType(base=f"Element_{i}[{typ.base}]"),
                    env,
                )
        elif isinstance(target, ast.Starred):
            self._bind_target(
                target.value,
                RefinementType(base=f"list[{typ.base}]"),
                env,
            )

    def _contains_yield(self, body: List[ast.stmt]) -> bool:
        """Check if any statement in body contains yield/yield from."""
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Yield, ast.YieldFrom)):
                    return True
        return False

    def _extract_return_annotation(
        self, node: ast.AsyncFunctionDef
    ) -> RefinementType:
        """Extract the return type annotation."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return RefinementType(base=node.returns.id)
            if isinstance(node.returns, ast.Constant):
                return RefinementType(base=str(node.returns.value))
            return RefinementType(base=ast.dump(node.returns))
        return UNKNOWN_TYPE

    @property
    def in_async_context(self) -> bool:
        """Whether we are currently inside an async function."""
        return self._in_async_context


# ===========================================================================
# ClosureAnalyzer
# ===========================================================================

class ClosureAnalyzer:
    """Analyze nested functions: free variable capture, nonlocal tracking,
    closure refinement propagation, and mutable state detection."""

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._diagnostics: List[Diagnostic] = []
        self._closure_stack: Deque[ClosureScope] = deque()

    def analyze_nested_function(
        self, node: ast.FunctionDef, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a nested function definition and its closures."""
        loc = Location.from_node(node, self._file)

        local_names = self._collect_local_names(node)
        nonlocal_names = self._collect_nonlocal_names(node)
        global_names = self._collect_global_names(node)

        all_referenced = self._collect_referenced_names(node)
        free_vars = (
            all_referenced - local_names - nonlocal_names - global_names
        )

        captured: Dict[str, CapturedVar] = {}
        for name in free_vars:
            outer_type = env.lookup(name)
            if outer_type is not None:
                is_mutable = self._is_mutated_in(name, node.body)
                captured[name] = CapturedVar(
                    name=name,
                    outer_type=outer_type,
                    is_mutable=is_mutable,
                    is_nonlocal=name in nonlocal_names,
                )

        for name in nonlocal_names:
            outer_type = env.lookup(name)
            if outer_type is None:
                self._diagnostics.append(Diagnostic(
                    message=f"nonlocal '{name}' not found in enclosing scope",
                    location=loc,
                    severity="error",
                ))
            else:
                is_mutable = self._is_mutated_in(name, node.body)
                captured[name] = CapturedVar(
                    name=name,
                    outer_type=outer_type,
                    is_mutable=True,
                    is_nonlocal=True,
                )

        scope = ClosureScope(
            func_name=node.name,
            captured=captured,
            local_names=local_names,
            nonlocal_names=nonlocal_names,
            global_names=global_names,
            free_vars=free_vars,
            location=loc,
        )
        self._closure_stack.append(scope)

        func_env = env.child(ScopeKind.FUNCTION)
        for name, cap in captured.items():
            func_env.bind(name, cap.outer_type)
        self._bind_params(node.args, func_env)

        self._detect_mutable_capture_warnings(scope)

        func_type = RefinementType(
            base=f"Callable[..., Unknown]",
            location=loc,
        )
        if captured:
            func_type = func_type.narrow(
                RefinementKind.CALLABLE,
                {"captures": list(captured.keys())},
            )
        env.bind(node.name, func_type)

        return AnalysisResult(
            typ=func_type, env=env, diagnostics=self._diagnostics[:]
        )

    def get_closure_info(self, func_name: str) -> Optional[ClosureScope]:
        """Look up closure information for a named function."""
        for scope in reversed(self._closure_stack):
            if scope.func_name == func_name:
                return scope
        return None

    # -- internal helpers ---------------------------------------------------

    def _collect_local_names(self, node: ast.FunctionDef) -> Set[str]:
        """Collect names assigned locally in the function."""
        names: Set[str] = set()
        for arg in node.args.posonlyargs + node.args.args:
            names.add(arg.arg)
        if node.args.vararg:
            names.add(node.args.vararg.arg)
        for arg in node.args.kwonlyargs:
            names.add(arg.arg)
        if node.args.kwarg:
            names.add(node.args.kwarg.arg)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, ast.FunctionDef):
                if child is not node:
                    names.add(child.name)
            elif isinstance(child, ast.AsyncFunctionDef):
                names.add(child.name)
            elif isinstance(child, ast.ClassDef):
                names.add(child.name)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    names.add(alias.asname or alias.name)
        return names

    def _collect_nonlocal_names(self, node: ast.FunctionDef) -> Set[str]:
        """Collect names declared nonlocal in the function."""
        names: Set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Nonlocal):
                names.update(stmt.names)
            for child in ast.walk(stmt):
                if isinstance(child, ast.Nonlocal):
                    names.update(child.names)
        return names

    def _collect_global_names(self, node: ast.FunctionDef) -> Set[str]:
        """Collect names declared global in the function."""
        names: Set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Global):
                names.update(stmt.names)
            for child in ast.walk(stmt):
                if isinstance(child, ast.Global):
                    names.update(child.names)
        return names

    def _collect_referenced_names(self, node: ast.FunctionDef) -> Set[str]:
        """Collect all names referenced (loaded) inside the function."""
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
        return names

    def _is_mutated_in(self, name: str, body: List[ast.stmt]) -> bool:
        """Check if a variable name is assigned to in a list of statements."""
        for stmt in body:
            for child in ast.walk(stmt):
                if (
                    isinstance(child, ast.Name)
                    and child.id == name
                    and isinstance(child.ctx, ast.Store)
                ):
                    return True
                if isinstance(child, ast.AugAssign):
                    if (
                        isinstance(child.target, ast.Name)
                        and child.target.id == name
                    ):
                        return True
        return False

    def _detect_mutable_capture_warnings(self, scope: ClosureScope) -> None:
        """Emit warnings for closures capturing mutable variables."""
        for name, cap in scope.captured.items():
            if cap.is_mutable and not cap.is_nonlocal:
                self._diagnostics.append(Diagnostic(
                    message=(
                        f"Closure '{scope.func_name}' captures mutable "
                        f"variable '{name}' without nonlocal declaration"
                    ),
                    location=scope.location,
                    severity="warning",
                ))

    def _bind_params(self, args: ast.arguments, env: TypeEnv) -> None:
        """Bind function parameters in the environment."""
        for arg in args.posonlyargs + args.args:
            env.bind(arg.arg, UNKNOWN_TYPE)
        if args.vararg:
            env.bind(args.vararg.arg, RefinementType(base="tuple[Unknown, ...]"))
        for arg in args.kwonlyargs:
            env.bind(arg.arg, UNKNOWN_TYPE)
        if args.kwarg:
            env.bind(args.kwarg.arg, RefinementType(base="dict[str, Unknown]"))


@dataclass
class CapturedVar:
    """Information about a captured variable in a closure."""
    name: str
    outer_type: RefinementType
    is_mutable: bool
    is_nonlocal: bool


@dataclass
class ClosureScope:
    """Closure scope information for a nested function."""
    func_name: str
    captured: Dict[str, CapturedVar]
    local_names: Set[str]
    nonlocal_names: Set[str]
    global_names: Set[str]
    free_vars: Set[str]
    location: Location


# ===========================================================================
# PatternMatchAnalyzer
# ===========================================================================

class PatternMatchAnalyzer:
    """Analyze Python 3.10+ match/case statements with refinement
    type propagation through structural pattern matching."""

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._diagnostics: List[Diagnostic] = []

    def analyze_match(
        self, node: ast.AST, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a match statement (Python 3.10+)."""
        loc = Location.from_node(node, self._file)

        subject = getattr(node, "subject", None)
        cases = getattr(node, "cases", [])

        if subject is None:
            return AnalysisResult(typ=NONE_TYPE, env=env)

        subject_type = self._infer_subject_type(subject, env)
        case_results: List[AnalysisResult] = []
        covered_patterns: List[PatternKind] = []

        for case in cases:
            pattern = getattr(case, "pattern", None)
            guard = getattr(case, "guard", None)
            body = getattr(case, "body", [])

            case_env = env.child(ScopeKind.MATCH_CASE)

            if pattern is not None:
                pattern_info = self._analyze_pattern(
                    pattern, subject_type, case_env
                )
                covered_patterns.append(pattern_info.kind)

            if guard is not None:
                case_env = self._apply_guard(guard, case_env)

            case_result = self._analyze_case_body(body, case_env)
            case_results.append(case_result)

        is_exhaustive = self._check_exhaustiveness(
            covered_patterns, subject_type
        )
        if not is_exhaustive:
            self._diagnostics.append(Diagnostic(
                message="Match statement may not be exhaustive",
                location=loc,
                severity="warning",
            ))

        reachable = any(r.reachable for r in case_results) or not is_exhaustive

        return AnalysisResult(
            typ=NONE_TYPE,
            env=env,
            effects=[f"match@{loc.line}"],
            diagnostics=self._diagnostics[:],
            reachable=reachable,
        )

    def _analyze_pattern(
        self,
        pattern: ast.AST,
        subject_type: RefinementType,
        env: TypeEnv,
    ) -> PatternInfo:
        """Analyze a single pattern and bind variables."""
        class_name = type(pattern).__name__

        if class_name == "MatchValue":
            return self._analyze_value_pattern(pattern, subject_type, env)
        elif class_name == "MatchSingleton":
            return self._analyze_singleton_pattern(pattern, subject_type, env)
        elif class_name == "MatchSequence":
            return self._analyze_sequence_pattern(pattern, subject_type, env)
        elif class_name == "MatchMapping":
            return self._analyze_mapping_pattern(pattern, subject_type, env)
        elif class_name == "MatchClass":
            return self._analyze_class_pattern(pattern, subject_type, env)
        elif class_name == "MatchStar":
            return self._analyze_star_pattern(pattern, subject_type, env)
        elif class_name == "MatchAs":
            return self._analyze_as_pattern(pattern, subject_type, env)
        elif class_name == "MatchOr":
            return self._analyze_or_pattern(pattern, subject_type, env)
        else:
            return PatternInfo(kind=PatternKind.WILDCARD, bindings={})

    def _analyze_value_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchValue pattern (matches by equality)."""
        value_node = getattr(pattern, "value", None)
        if value_node is not None and isinstance(value_node, ast.Constant):
            lit_type = _LITERAL_MAP.get(type(value_node.value), UNKNOWN_TYPE)
            refined = subject_type.narrow(
                RefinementKind.LITERAL, value_node.value
            )
            return PatternInfo(kind=PatternKind.LITERAL, bindings={}, narrowed_type=refined)
        return PatternInfo(kind=PatternKind.VALUE, bindings={})

    def _analyze_singleton_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchSingleton pattern (True, False, None)."""
        value = getattr(pattern, "value", None)
        if value is None:
            return PatternInfo(
                kind=PatternKind.LITERAL,
                bindings={},
                narrowed_type=NONE_TYPE,
            )
        if value is True:
            return PatternInfo(
                kind=PatternKind.LITERAL,
                bindings={},
                narrowed_type=BOOL_TYPE.narrow(RefinementKind.LITERAL, True),
            )
        if value is False:
            return PatternInfo(
                kind=PatternKind.LITERAL,
                bindings={},
                narrowed_type=BOOL_TYPE.narrow(RefinementKind.LITERAL, False),
            )
        return PatternInfo(kind=PatternKind.LITERAL, bindings={})

    def _analyze_sequence_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchSequence pattern ([p1, p2, ...])."""
        patterns = getattr(pattern, "patterns", [])
        bindings: Dict[str, RefinementType] = {}
        for i, sub_pat in enumerate(patterns):
            elem_type = RefinementType(base=f"Element_{i}[{subject_type.base}]")
            sub_info = self._analyze_pattern(sub_pat, elem_type, env)
            bindings.update(sub_info.bindings)
            for name, typ in sub_info.bindings.items():
                env.bind(name, typ)

        narrowed = subject_type.narrow(
            RefinementKind.LENGTH, len(patterns)
        )
        return PatternInfo(
            kind=PatternKind.SEQUENCE,
            bindings=bindings,
            narrowed_type=narrowed,
        )

    def _analyze_mapping_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchMapping pattern ({k1: p1, k2: p2, **rest})."""
        keys = getattr(pattern, "keys", [])
        kw_patterns = getattr(pattern, "patterns", [])
        rest = getattr(pattern, "rest", None)
        bindings: Dict[str, RefinementType] = {}

        for key, val_pat in zip(keys, kw_patterns):
            val_type = RefinementType(base=f"ValueOf[{subject_type.base}]")
            sub_info = self._analyze_pattern(val_pat, val_type, env)
            bindings.update(sub_info.bindings)
            for name, typ in sub_info.bindings.items():
                env.bind(name, typ)

        if rest is not None:
            rest_type = RefinementType(base=f"dict[str, Unknown]")
            env.bind(rest, rest_type)
            bindings[rest] = rest_type

        return PatternInfo(kind=PatternKind.MAPPING, bindings=bindings)

    def _analyze_class_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchClass pattern (ClassName(p1, k=p2))."""
        cls_node = getattr(pattern, "cls", None)
        patterns = getattr(pattern, "patterns", [])
        kwd_attrs = getattr(pattern, "kwd_attrs", [])
        kwd_patterns = getattr(pattern, "kwd_patterns", [])
        bindings: Dict[str, RefinementType] = {}

        if cls_node is not None:
            cls_name = self._pattern_class_name(cls_node)
            narrowed = subject_type.narrow(RefinementKind.ISINSTANCE, cls_name)
        else:
            cls_name = "Unknown"
            narrowed = subject_type

        for i, sub_pat in enumerate(patterns):
            elem_type = RefinementType(base=f"PosArg_{i}[{cls_name}]")
            sub_info = self._analyze_pattern(sub_pat, elem_type, env)
            bindings.update(sub_info.bindings)
            for name, typ in sub_info.bindings.items():
                env.bind(name, typ)

        for attr_name, kwd_pat in zip(kwd_attrs, kwd_patterns):
            attr_type = RefinementType(base=f"Attr_{attr_name}[{cls_name}]")
            sub_info = self._analyze_pattern(kwd_pat, attr_type, env)
            bindings.update(sub_info.bindings)
            for name, typ in sub_info.bindings.items():
                env.bind(name, typ)

        return PatternInfo(
            kind=PatternKind.CLASS,
            bindings=bindings,
            narrowed_type=narrowed,
        )

    def _analyze_star_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchStar pattern (*name)."""
        name = getattr(pattern, "name", None)
        bindings: Dict[str, RefinementType] = {}
        if name is not None:
            rest_type = RefinementType(base=f"list[ElementOf[{subject_type.base}]]")
            env.bind(name, rest_type)
            bindings[name] = rest_type
        return PatternInfo(kind=PatternKind.STAR, bindings=bindings)

    def _analyze_as_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchAs pattern (pattern as name / wildcard)."""
        sub_pattern = getattr(pattern, "pattern", None)
        name = getattr(pattern, "name", None)
        bindings: Dict[str, RefinementType] = {}

        if sub_pattern is not None:
            sub_info = self._analyze_pattern(sub_pattern, subject_type, env)
            bindings.update(sub_info.bindings)
            narrowed = sub_info.narrowed_type or subject_type
        else:
            narrowed = subject_type

        if name is not None:
            env.bind(name, narrowed)
            bindings[name] = narrowed

        kind = PatternKind.AS if sub_pattern is not None else PatternKind.WILDCARD
        return PatternInfo(kind=kind, bindings=bindings, narrowed_type=narrowed)

    def _analyze_or_pattern(
        self, pattern: ast.AST, subject_type: RefinementType, env: TypeEnv
    ) -> PatternInfo:
        """Analyze a MatchOr pattern (p1 | p2 | p3)."""
        sub_patterns = getattr(pattern, "patterns", [])
        all_bindings: Dict[str, RefinementType] = {}
        narrowed_types: List[RefinementType] = []

        for sub_pat in sub_patterns:
            branch_env = env.child(ScopeKind.MATCH_CASE)
            sub_info = self._analyze_pattern(sub_pat, subject_type, branch_env)
            for name, typ in sub_info.bindings.items():
                if name in all_bindings:
                    all_bindings[name] = all_bindings[name].join(typ)
                else:
                    all_bindings[name] = typ
            if sub_info.narrowed_type is not None:
                narrowed_types.append(sub_info.narrowed_type)

        for name, typ in all_bindings.items():
            env.bind(name, typ)

        combined_narrow = None
        if narrowed_types:
            combined_narrow = narrowed_types[0]
            for t in narrowed_types[1:]:
                combined_narrow = combined_narrow.join(t)

        return PatternInfo(
            kind=PatternKind.OR,
            bindings=all_bindings,
            narrowed_type=combined_narrow,
        )

    def _apply_guard(self, guard: ast.AST, env: TypeEnv) -> TypeEnv:
        """Apply refinements from a case guard condition."""
        if isinstance(guard, ast.Compare):
            left = guard.left
            for op, comp in zip(guard.ops, guard.comparators):
                if isinstance(left, ast.Name):
                    existing = env.lookup(left.id) or UNKNOWN_TYPE
                    if isinstance(op, ast.IsNot) and isinstance(
                        comp, ast.Constant
                    ) and comp.value is None:
                        env.bind(
                            left.id,
                            existing.narrow(RefinementKind.NOTNONE, True),
                        )
                left = comp
        elif isinstance(guard, ast.Call):
            if (
                isinstance(guard.func, ast.Name)
                and guard.func.id == "isinstance"
                and len(guard.args) == 2
                and isinstance(guard.args[0], ast.Name)
            ):
                name = guard.args[0].id
                type_name = self._pattern_class_name(guard.args[1])
                existing = env.lookup(name) or UNKNOWN_TYPE
                env.bind(
                    name,
                    existing.narrow(RefinementKind.ISINSTANCE, type_name),
                )
        return env

    def _analyze_case_body(
        self, body: List[ast.stmt], env: TypeEnv
    ) -> AnalysisResult:
        """Analyze the body of a case clause."""
        reachable = True
        for stmt in body:
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break)):
                reachable = False
                break
        return AnalysisResult(typ=NONE_TYPE, env=env, reachable=reachable)

    def _infer_subject_type(
        self, node: ast.AST, env: TypeEnv
    ) -> RefinementType:
        """Infer the type of the match subject expression."""
        if isinstance(node, ast.Name):
            return env.lookup(node.id) or UNKNOWN_TYPE
        if isinstance(node, ast.Constant):
            return _LITERAL_MAP.get(type(node.value), UNKNOWN_TYPE)
        if isinstance(node, ast.Tuple):
            elem_types = [
                self._infer_subject_type(e, env).base for e in node.elts
            ]
            return RefinementType(base=f"tuple[{', '.join(elem_types)}]")
        return UNKNOWN_TYPE

    def _pattern_class_name(self, node: ast.AST) -> str:
        """Extract a class name from a pattern node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._pattern_class_name(node.value)}.{node.attr}"
        return "Unknown"

    def _check_exhaustiveness(
        self,
        covered: List[PatternKind],
        subject_type: RefinementType,
    ) -> bool:
        """Check if pattern cases cover all possible values."""
        if PatternKind.WILDCARD in covered:
            return True
        if PatternKind.CAPTURE in covered:
            return True
        if subject_type.base == "bool" and covered.count(PatternKind.LITERAL) >= 2:
            return True
        if subject_type.base.startswith("Union["):
            variants = subject_type.base[6:-1].split(",")
            if len(covered) >= len(variants):
                return True
        return False


@dataclass
class PatternInfo:
    """Result of analyzing a single pattern."""
    kind: PatternKind
    bindings: Dict[str, RefinementType]
    narrowed_type: Optional[RefinementType] = None


# ===========================================================================
# FullPythonAnalyzer — main entry point
# ===========================================================================

class FullPythonAnalyzer:
    """Main analyzer that composes all sub-analyzers and walks the full AST.

    Dispatches to specialized analyzers for each construct category and
    manages the global typing environment with proper scoping.
    """

    def __init__(self, file: str = "<unknown>") -> None:
        self._file = file
        self._comprehension = ComprehensionAnalyzer(file)
        self._class = ClassAnalyzer(file)
        self._context = ContextManagerAnalyzer(file)
        self._exception = ExceptionFlowAnalyzer(file)
        self._async = AsyncAnalyzer(file)
        self._closure = ClosureAnalyzer(file)
        self._pattern = PatternMatchAnalyzer(file)
        self._diagnostics: List[Diagnostic] = []
        self._env = TypeEnv(scope_kind=ScopeKind.MODULE)

    def analyze_module(self, source: str) -> AnalysisResult:
        """Parse and analyze an entire Python module from source text."""
        try:
            tree = ast.parse(source, filename=self._file)
        except SyntaxError as e:
            return AnalysisResult(
                typ=NONE_TYPE,
                env=self._env,
                diagnostics=[Diagnostic(
                    message=f"Syntax error: {e}",
                    location=Location(self._file, e.lineno or 0, e.offset or 0),
                    severity="error",
                )],
            )
        return self.analyze_ast(tree)

    def analyze_ast(self, tree: ast.Module) -> AnalysisResult:
        """Analyze a pre-parsed AST module node."""
        for stmt in tree.body:
            self._analyze_stmt(stmt, self._env)
        return AnalysisResult(
            typ=NONE_TYPE,
            env=self._env,
            diagnostics=self._collect_all_diagnostics(),
        )

    def _analyze_stmt(self, stmt: ast.AST, env: TypeEnv) -> AnalysisResult:
        """Dispatch a statement to the appropriate handler."""
        if isinstance(stmt, ast.FunctionDef):
            return self._analyze_funcdef(stmt, env)
        if isinstance(stmt, ast.AsyncFunctionDef):
            return self._async.analyze_async_funcdef(stmt, env)
        if isinstance(stmt, ast.ClassDef):
            return self._class.analyze_classdef(stmt, env)
        if isinstance(stmt, ast.Return):
            return self._analyze_return(stmt, env)
        if isinstance(stmt, ast.Assign):
            return self._analyze_assign(stmt, env)
        if isinstance(stmt, ast.AugAssign):
            return self._analyze_augassign(stmt, env)
        if isinstance(stmt, ast.AnnAssign):
            return self._analyze_annassign(stmt, env)
        if isinstance(stmt, ast.For):
            return self._analyze_for(stmt, env)
        if isinstance(stmt, ast.AsyncFor):
            return self._async.analyze_async_for(stmt, env)
        if isinstance(stmt, ast.While):
            return self._analyze_while(stmt, env)
        if isinstance(stmt, ast.If):
            return self._analyze_if(stmt, env)
        if isinstance(stmt, ast.With):
            return self._context.analyze_with(stmt, env)
        if isinstance(stmt, ast.AsyncWith):
            return self._context.analyze_async_with(stmt, env)
        if isinstance(stmt, ast.Raise):
            return self._exception.analyze_raise(stmt, env)
        if isinstance(stmt, ast.Try):
            return self._exception.analyze_try(stmt, env)
        if isinstance(stmt, ast.Assert):
            return self._analyze_assert(stmt, env)
        if isinstance(stmt, ast.Import):
            return self._analyze_import(stmt, env)
        if isinstance(stmt, ast.ImportFrom):
            return self._analyze_import_from(stmt, env)
        if isinstance(stmt, ast.Global):
            return AnalysisResult(typ=NONE_TYPE, env=env)
        if isinstance(stmt, ast.Nonlocal):
            return AnalysisResult(typ=NONE_TYPE, env=env)
        if isinstance(stmt, ast.Expr):
            return self._analyze_expr_stmt(stmt, env)
        if isinstance(stmt, ast.Pass):
            return AnalysisResult(typ=NONE_TYPE, env=env)
        if isinstance(stmt, ast.Break):
            return AnalysisResult(typ=NONE_TYPE, env=env, reachable=False)
        if isinstance(stmt, ast.Continue):
            return AnalysisResult(typ=NONE_TYPE, env=env, reachable=False)
        if isinstance(stmt, ast.Delete):
            return self._analyze_delete(stmt, env)
        # Python 3.10+ match
        if type(stmt).__name__ == "Match":
            return self._pattern.analyze_match(stmt, env)
        # Python 3.11+ try/except*
        if type(stmt).__name__ == "TryStar":
            return self._exception.analyze_try_star(stmt, env)
        return AnalysisResult(typ=UNKNOWN_TYPE, env=env)

    def _analyze_funcdef(
        self, node: ast.FunctionDef, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a (sync) function definition."""
        loc = Location.from_node(node, self._file)

        is_nested = env.scope_kind == ScopeKind.FUNCTION
        if is_nested:
            return self._closure.analyze_nested_function(node, env)

        func_env = env.child(ScopeKind.FUNCTION)
        self._bind_params(node.args, func_env)

        for stmt in node.body:
            self._analyze_stmt(stmt, func_env)

        return_type = UNKNOWN_TYPE
        if node.returns:
            return_type = self._annotation_to_type(node.returns)

        func_type = RefinementType(
            base=f"Callable[..., {return_type.base}]",
            location=loc,
        )
        env.bind(node.name, func_type)
        return AnalysisResult(typ=func_type, env=env)

    def _analyze_return(
        self, node: ast.Return, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a return statement."""
        if node.value is not None:
            ret_type = self._analyze_expr(node.value, env)
        else:
            ret_type = NONE_TYPE
        return AnalysisResult(typ=ret_type, env=env, reachable=False)

    def _analyze_assign(
        self, node: ast.Assign, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an assignment statement."""
        val_type = self._analyze_expr(node.value, env)
        for target in node.targets:
            self._bind_assignment_target(target, val_type, env)
        return AnalysisResult(typ=val_type, env=env)

    def _analyze_augassign(
        self, node: ast.AugAssign, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an augmented assignment (+=, -=, etc.)."""
        val_type = self._analyze_expr(node.value, env)
        if isinstance(node.target, ast.Name):
            existing = env.lookup(node.target.id) or UNKNOWN_TYPE
            env.bind(node.target.id, existing.join(val_type))
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_annassign(
        self, node: ast.AnnAssign, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an annotated assignment."""
        ann_type = self._annotation_to_type(node.annotation)
        if isinstance(node.target, ast.Name):
            env.bind(node.target.id, ann_type)
        if node.value is not None:
            self._analyze_expr(node.value, env)
        return AnalysisResult(typ=ann_type, env=env)

    def _analyze_for(
        self, node: ast.For, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a for loop."""
        iter_type = self._analyze_expr(node.iter, env)
        elem_type = RefinementType(base=f"ElementOf[{iter_type.base}]")
        self._bind_assignment_target(node.target, elem_type, env)
        for stmt in node.body:
            self._analyze_stmt(stmt, env)
        for stmt in node.orelse:
            self._analyze_stmt(stmt, env)
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_while(
        self, node: ast.While, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a while loop."""
        self._analyze_expr(node.test, env)
        for stmt in node.body:
            self._analyze_stmt(stmt, env)
        for stmt in node.orelse:
            self._analyze_stmt(stmt, env)
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_if(
        self, node: ast.If, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an if/elif/else statement with refinement narrowing."""
        test_type = self._analyze_expr(node.test, env)

        then_env = env.child(env.scope_kind)
        self._apply_condition_refinements(node.test, then_env, positive=True)
        for stmt in node.body:
            self._analyze_stmt(stmt, then_env)

        else_env = env.child(env.scope_kind)
        self._apply_condition_refinements(node.test, else_env, positive=False)
        for stmt in node.orelse:
            self._analyze_stmt(stmt, else_env)

        self._merge_branch_envs(env, then_env, else_env)
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_assert(
        self, node: ast.Assert, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an assert statement (applies positive refinement)."""
        self._apply_condition_refinements(node.test, env, positive=True)
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_import(
        self, node: ast.Import, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze an import statement."""
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            env.bind(name, RefinementType(base=f"module[{alias.name}]"))
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_import_from(
        self, node: ast.ImportFrom, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a from-import statement."""
        mod = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            env.bind(name, RefinementType(base=f"imported[{mod}.{alias.name}]"))
        return AnalysisResult(typ=NONE_TYPE, env=env)

    def _analyze_expr_stmt(
        self, node: ast.Expr, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a bare expression statement."""
        typ = self._analyze_expr(node.value, env)
        return AnalysisResult(typ=typ, env=env)

    def _analyze_delete(
        self, node: ast.Delete, env: TypeEnv
    ) -> AnalysisResult:
        """Analyze a del statement."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                env.bind(target.id, NEVER_TYPE)
        return AnalysisResult(typ=NONE_TYPE, env=env)

    # -- expression analysis ------------------------------------------------

    def _analyze_expr(self, node: ast.AST, env: TypeEnv) -> RefinementType:
        """Analyze an expression and return its refinement type."""
        if isinstance(node, ast.Constant):
            return _LITERAL_MAP.get(type(node.value), UNKNOWN_TYPE)
        if isinstance(node, ast.Name):
            return env.lookup(node.id) or UNKNOWN_TYPE
        if isinstance(node, ast.BinOp):
            left = self._analyze_expr(node.left, env)
            right = self._analyze_expr(node.right, env)
            return left.join(right)
        if isinstance(node, ast.UnaryOp):
            return self._analyze_expr(node.operand, env)
        if isinstance(node, ast.BoolOp):
            types = [self._analyze_expr(v, env) for v in node.values]
            result = types[0]
            for t in types[1:]:
                result = result.join(t)
            return result
        if isinstance(node, ast.Compare):
            self._analyze_expr(node.left, env)
            for comp in node.comparators:
                self._analyze_expr(comp, env)
            return BOOL_TYPE
        if isinstance(node, ast.Call):
            return self._analyze_call(node, env)
        if isinstance(node, ast.Attribute):
            self._analyze_expr(node.value, env)
            return UNKNOWN_TYPE
        if isinstance(node, ast.Subscript):
            self._analyze_expr(node.value, env)
            return UNKNOWN_TYPE
        if isinstance(node, ast.IfExp):
            then_type = self._analyze_expr(node.body, env)
            else_type = self._analyze_expr(node.orelse, env)
            return then_type.join(else_type)
        if isinstance(node, ast.Lambda):
            return RefinementType(base="Callable[..., Unknown]")
        if isinstance(node, ast.ListComp):
            return self._comprehension.analyze_listcomp(node, env).typ
        if isinstance(node, ast.SetComp):
            return self._comprehension.analyze_setcomp(node, env).typ
        if isinstance(node, ast.DictComp):
            return self._comprehension.analyze_dictcomp(node, env).typ
        if isinstance(node, ast.GeneratorExp):
            return self._comprehension.analyze_generatorexp(node, env).typ
        if isinstance(node, ast.Await):
            return self._async.analyze_await(node, env).typ
        if isinstance(node, ast.Yield):
            return self._async.analyze_yield(node, env).typ
        if isinstance(node, ast.YieldFrom):
            return self._async.analyze_yield_from(node, env).typ
        if isinstance(node, ast.List):
            elem_types = [self._analyze_expr(e, env) for e in node.elts]
            if elem_types:
                inner = elem_types[0]
                for t in elem_types[1:]:
                    inner = inner.join(t)
                return RefinementType(base=f"list[{inner.base}]")
            return LIST_TYPE
        if isinstance(node, ast.Dict):
            return DICT_TYPE
        if isinstance(node, ast.Set):
            return SET_TYPE
        if isinstance(node, ast.Tuple):
            elem_types = [self._analyze_expr(e, env).base for e in node.elts]
            return RefinementType(base=f"tuple[{', '.join(elem_types)}]")
        if isinstance(node, ast.FormattedValue):
            return STR_TYPE
        if isinstance(node, ast.JoinedStr):
            return STR_TYPE
        if isinstance(node, ast.Starred):
            inner = self._analyze_expr(node.value, env)
            return RefinementType(base=f"Unpacked[{inner.base}]")
        # Walrus operator (:=)
        if isinstance(node, ast.NamedExpr):
            val_type = self._analyze_expr(node.value, env)
            if isinstance(node.target, ast.Name):
                env.bind(node.target.id, val_type)
            return val_type
        if isinstance(node, ast.Slice):
            return RefinementType(base="slice")
        return UNKNOWN_TYPE

    def _analyze_call(
        self, node: ast.Call, env: TypeEnv
    ) -> RefinementType:
        """Analyze a function call expression."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            func_type = env.lookup(func_name)
            if func_type is not None and func_type.base.startswith("Callable["):
                parts = func_type.base.split(",")
                ret_part = parts[-1].strip().rstrip("]")
                return RefinementType(base=ret_part)
            if func_name in ("int", "float", "str", "bool", "bytes", "list",
                             "dict", "set", "tuple", "frozenset"):
                return RefinementType(base=func_name)
            if func_name == "len":
                return INT_TYPE.narrow(RefinementKind.RANGE, (">=", 0))
            if func_name == "type":
                return RefinementType(base="type")
            if func_name == "isinstance":
                return BOOL_TYPE
            if func_name == "hasattr":
                return BOOL_TYPE
            return RefinementType(base=f"ReturnOf[{func_name}]")
        if isinstance(node.func, ast.Attribute):
            return RefinementType(base=f"ReturnOf[_.{node.func.attr}]")
        return UNKNOWN_TYPE

    # -- refinement propagation helpers -------------------------------------

    def _apply_condition_refinements(
        self, node: ast.AST, env: TypeEnv, *, positive: bool
    ) -> None:
        """Apply refinements from a condition to the environment."""
        if isinstance(node, ast.Compare):
            self._apply_compare_refinements(node, env, positive=positive)
        elif isinstance(node, ast.Call):
            self._apply_call_refinements(node, env, positive=positive)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._apply_condition_refinements(
                node.operand, env, positive=not positive
            )
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And) and positive:
                for val in node.values:
                    self._apply_condition_refinements(val, env, positive=True)
            elif isinstance(node.op, ast.Or) and not positive:
                for val in node.values:
                    self._apply_condition_refinements(val, env, positive=False)
        elif isinstance(node, ast.Name) and positive:
            existing = env.lookup(node.id)
            if existing is not None:
                env.bind(
                    node.id,
                    existing.narrow(RefinementKind.TRUTHINESS, True),
                )

    def _apply_compare_refinements(
        self, node: ast.Compare, env: TypeEnv, *, positive: bool
    ) -> None:
        """Apply refinements from comparisons."""
        left = node.left
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(left, ast.Name):
                existing = env.lookup(left.id) or UNKNOWN_TYPE
                if positive:
                    if isinstance(op, ast.Is) and isinstance(comp, ast.Constant) and comp.value is None:
                        env.bind(left.id, NONE_TYPE)
                    elif isinstance(op, ast.IsNot) and isinstance(comp, ast.Constant) and comp.value is None:
                        env.bind(left.id, existing.narrow(RefinementKind.NOTNONE, True))
                    elif isinstance(op, (ast.Eq,)) and isinstance(comp, ast.Constant):
                        env.bind(left.id, existing.narrow(RefinementKind.LITERAL, comp.value))
                else:
                    if isinstance(op, ast.Is) and isinstance(comp, ast.Constant) and comp.value is None:
                        env.bind(left.id, existing.narrow(RefinementKind.NOTNONE, True))
                    elif isinstance(op, ast.IsNot) and isinstance(comp, ast.Constant) and comp.value is None:
                        env.bind(left.id, NONE_TYPE)
            left = comp

    def _apply_call_refinements(
        self, node: ast.Call, env: TypeEnv, *, positive: bool
    ) -> None:
        """Apply refinements from call-based conditions (isinstance, etc.)."""
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
        ):
            target = node.args[0]
            type_arg = node.args[1]
            if isinstance(target, ast.Name):
                type_name = self._extract_type_name(type_arg)
                existing = env.lookup(target.id) or UNKNOWN_TYPE
                if positive:
                    env.bind(
                        target.id,
                        existing.narrow(RefinementKind.ISINSTANCE, type_name),
                    )
                else:
                    env.bind(
                        target.id,
                        existing.narrow(
                            RefinementKind.NEGATION,
                            f"not isinstance {type_name}",
                        ),
                    )

    def _extract_type_name(self, node: ast.AST) -> str:
        """Extract type name from isinstance second arg."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Tuple):
            parts = [self._extract_type_name(e) for e in node.elts]
            return f"Union[{', '.join(parts)}]"
        if isinstance(node, ast.Attribute):
            return f"{self._extract_type_name(node.value)}.{node.attr}"
        return "Unknown"

    # -- binding helpers ----------------------------------------------------

    def _bind_assignment_target(
        self, target: ast.AST, typ: RefinementType, env: TypeEnv
    ) -> None:
        """Bind an assignment target (possibly destructured)."""
        if isinstance(target, ast.Name):
            env.bind(target.id, typ)
        elif isinstance(target, ast.Tuple):
            for i, elt in enumerate(target.elts):
                elem_type = RefinementType(base=f"Element_{i}[{typ.base}]")
                self._bind_assignment_target(elt, elem_type, env)
        elif isinstance(target, ast.List):
            for i, elt in enumerate(target.elts):
                elem_type = RefinementType(base=f"Element_{i}[{typ.base}]")
                self._bind_assignment_target(elt, elem_type, env)
        elif isinstance(target, ast.Starred):
            rest_type = RefinementType(base=f"list[{typ.base}]")
            self._bind_assignment_target(target.value, rest_type, env)

    def _bind_params(self, args: ast.arguments, env: TypeEnv) -> None:
        """Bind function parameters in the environment."""
        for arg in args.posonlyargs + args.args:
            if arg.annotation:
                env.bind(arg.arg, self._annotation_to_type(arg.annotation))
            else:
                env.bind(arg.arg, UNKNOWN_TYPE)
        if args.vararg:
            env.bind(
                args.vararg.arg,
                RefinementType(base="tuple[Unknown, ...]"),
            )
        for arg in args.kwonlyargs:
            if arg.annotation:
                env.bind(arg.arg, self._annotation_to_type(arg.annotation))
            else:
                env.bind(arg.arg, UNKNOWN_TYPE)
        if args.kwarg:
            env.bind(
                args.kwarg.arg,
                RefinementType(base="dict[str, Unknown]"),
            )

    def _annotation_to_type(self, node: ast.AST) -> RefinementType:
        """Convert an annotation node to a RefinementType."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return RefinementType(base=node.value)
        if isinstance(node, ast.Name):
            return RefinementType(base=node.id)
        if isinstance(node, ast.Subscript):
            base = self._annotation_to_type(node.value).base
            return RefinementType(base=f"{base}[...]")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._annotation_to_type(node.left)
            right = self._annotation_to_type(node.right)
            return RefinementType(base=f"Union[{left.base}, {right.base}]")
        if isinstance(node, ast.Attribute):
            return RefinementType(base=f"{ast.dump(node)}")
        return UNKNOWN_TYPE

    def _merge_branch_envs(
        self, target: TypeEnv, then_env: TypeEnv, else_env: TypeEnv
    ) -> None:
        """Merge two branch environments back into the target."""
        all_names = set(then_env.bindings) | set(else_env.bindings)
        for name in all_names:
            then_type = then_env.bindings.get(name)
            else_type = else_env.bindings.get(name)
            if then_type is not None and else_type is not None:
                target.bind(name, then_type.join(else_type))
            elif then_type is not None:
                target.bind(name, then_type)
            elif else_type is not None:
                target.bind(name, else_type)

    # -- diagnostics --------------------------------------------------------

    def _collect_all_diagnostics(self) -> List[Diagnostic]:
        """Collect diagnostics from all sub-analyzers."""
        all_diags: List[Diagnostic] = list(self._diagnostics)
        all_diags.extend(self._comprehension._diagnostics)
        all_diags.extend(self._class._diagnostics)
        all_diags.extend(self._context._diagnostics)
        all_diags.extend(self._exception._diagnostics)
        all_diags.extend(self._async._diagnostics)
        all_diags.extend(self._closure._diagnostics)
        all_diags.extend(self._pattern._diagnostics)
        return all_diags

    @property
    def env(self) -> TypeEnv:
        """Current global typing environment."""
        return self._env

    @property
    def class_analyzer(self) -> ClassAnalyzer:
        """Access the class sub-analyzer."""
        return self._class

    @property
    def closure_analyzer(self) -> ClosureAnalyzer:
        """Access the closure sub-analyzer."""
        return self._closure

    @property
    def diagnostics(self) -> List[Diagnostic]:
        """All diagnostics accumulated so far."""
        return self._collect_all_diagnostics()
