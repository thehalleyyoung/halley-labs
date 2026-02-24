from __future__ import annotations

import ast
import builtins
import itertools
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
    Union,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScopeType(Enum):
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    COMPREHENSION = auto()
    LAMBDA = auto()
    TYPE_PARAM = auto()
    EXCEPTION = auto()


class BindingType(Enum):
    LOCAL = auto()
    GLOBAL = auto()
    NONLOCAL = auto()
    FREE = auto()
    CELL = auto()
    BUILTIN = auto()
    IMPLICIT_GLOBAL = auto()
    PARAMETER = auto()
    IMPORT = auto()
    ANNOTATION = auto()


class ParameterKind(Enum):
    POSITIONAL_ONLY = auto()
    POSITIONAL_OR_KEYWORD = auto()
    VAR_POSITIONAL = auto()
    KEYWORD_ONLY = auto()
    VAR_KEYWORD = auto()


class CaptureMode(Enum):
    BY_REFERENCE = auto()
    EXPLICIT_COPY = auto()


class DiagnosticSeverity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


# ---------------------------------------------------------------------------
# Location helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """A position in source code."""
    lineno: int
    col_offset: int
    end_lineno: Optional[int] = None
    end_col_offset: Optional[int] = None

    @classmethod
    def from_node(cls, node: ast.AST) -> SourceLocation:
        return cls(
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
        )

    def __repr__(self) -> str:
        return f"{self.lineno}:{self.col_offset}"


# ---------------------------------------------------------------------------
# UseSite / DefSite
# ---------------------------------------------------------------------------

@dataclass
class DefSite:
    """Records where a name is defined."""
    location: SourceLocation
    node: ast.AST
    kind: str = "assignment"  # assignment | parameter | import | ...

    def __repr__(self) -> str:
        return f"Def({self.kind}@{self.location})"


@dataclass
class UseSite:
    """Records where a name is used."""
    location: SourceLocation
    node: ast.AST
    context: str = "load"  # load | del | annotation

    def __repr__(self) -> str:
        return f"Use({self.context}@{self.location})"


# ---------------------------------------------------------------------------
# BindingInfo
# ---------------------------------------------------------------------------

@dataclass
class BindingInfo:
    """Information about a single variable binding inside a scope."""
    name: str
    scope: Optional[Scope] = None
    binding_type: BindingType = BindingType.LOCAL

    definition_sites: List[DefSite] = field(default_factory=list)
    use_sites: List[UseSite] = field(default_factory=list)

    is_mutable: bool = True
    is_annotated: bool = False
    annotation: Optional[ast.AST] = None
    is_parameter: bool = False
    parameter_kind: Optional[ParameterKind] = None
    default_value: Optional[ast.AST] = None

    is_imported: bool = False
    import_module: Optional[str] = None
    import_name: Optional[str] = None

    is_type_param: bool = False
    is_exception_var: bool = False
    is_walrus: bool = False
    is_match_var: bool = False
    is_comprehension_iter: bool = False

    def add_definition(self, node: ast.AST, kind: str = "assignment") -> None:
        loc = SourceLocation.from_node(node)
        self.definition_sites.append(DefSite(location=loc, node=node, kind=kind))

    def add_use(self, node: ast.AST, context: str = "load") -> None:
        loc = SourceLocation.from_node(node)
        self.use_sites.append(UseSite(location=loc, node=node, context=context))

    @property
    def is_defined(self) -> bool:
        return len(self.definition_sites) > 0

    @property
    def is_used(self) -> bool:
        return len(self.use_sites) > 0

    @property
    def is_single_assignment(self) -> bool:
        return len(self.definition_sites) == 1 and not self.is_mutable

    def merge(self, other: BindingInfo) -> None:
        self.definition_sites.extend(other.definition_sites)
        self.use_sites.extend(other.use_sites)
        if other.is_annotated:
            self.is_annotated = True
            self.annotation = other.annotation
        if other.is_parameter:
            self.is_parameter = True
            self.parameter_kind = other.parameter_kind
            self.default_value = other.default_value

    def __repr__(self) -> str:
        return (
            f"BindingInfo({self.name}, {self.binding_type.name}, "
            f"defs={len(self.definition_sites)}, uses={len(self.use_sites)})"
        )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

@dataclass
class Scope:
    """Represents a single lexical scope in Python source."""

    name: str
    scope_type: ScopeType
    node: ast.AST
    parent: Optional[Scope] = None
    children: List[Scope] = field(default_factory=list)

    bindings: Dict[str, BindingInfo] = field(default_factory=dict)

    free_vars: Set[str] = field(default_factory=set)
    cell_vars: Set[str] = field(default_factory=set)
    global_vars: Set[str] = field(default_factory=set)
    nonlocal_vars: Set[str] = field(default_factory=set)
    explicit_global_names: Set[str] = field(default_factory=set)
    explicit_nonlocal_names: Set[str] = field(default_factory=set)

    is_nested: bool = False
    is_generator: bool = False
    is_coroutine: bool = False
    is_comprehension_scope: bool = False

    _node_to_scope: Dict[int, Scope] = field(default_factory=dict, repr=False)
    _depth: int = 0

    # ------------------------------------------------------------------
    # Binding management
    # ------------------------------------------------------------------

    def add_binding(
        self,
        name: str,
        node: ast.AST,
        binding_type: BindingType = BindingType.LOCAL,
        kind: str = "assignment",
        **kwargs: Any,
    ) -> BindingInfo:
        if name in self.bindings:
            info = self.bindings[name]
            info.add_definition(node, kind=kind)
            for k, v in kwargs.items():
                if hasattr(info, k):
                    setattr(info, k, v)
            return info
        info = BindingInfo(name=name, scope=self, binding_type=binding_type, **kwargs)
        info.add_definition(node, kind=kind)
        self.bindings[name] = info
        return info

    def add_use(self, name: str, node: ast.AST, context: str = "load") -> None:
        if name in self.bindings:
            self.bindings[name].add_use(node, context)
        else:
            info = BindingInfo(name=name, scope=self, binding_type=BindingType.FREE)
            info.add_use(node, context)
            self.bindings[name] = info

    def lookup(self, name: str) -> Optional[BindingInfo]:
        return self.bindings.get(name)

    def define(
        self,
        name: str,
        node: ast.AST,
        binding_type: BindingType = BindingType.LOCAL,
        kind: str = "assignment",
    ) -> BindingInfo:
        return self.add_binding(name, node, binding_type, kind)

    def add_child(self, child: Scope) -> None:
        child.parent = self
        child._depth = self._depth + 1
        child.is_nested = self.scope_type in (
            ScopeType.FUNCTION,
            ScopeType.LAMBDA,
            ScopeType.COMPREHENSION,
        ) or self.is_nested
        self.children.append(child)

    def add_global(self, name: str) -> None:
        self.explicit_global_names.add(name)
        self.global_vars.add(name)
        if name in self.bindings:
            self.bindings[name].binding_type = BindingType.GLOBAL
        else:
            self.bindings[name] = BindingInfo(
                name=name, scope=self, binding_type=BindingType.GLOBAL
            )

    def add_nonlocal(self, name: str) -> None:
        self.explicit_nonlocal_names.add(name)
        self.nonlocal_vars.add(name)
        if name in self.bindings:
            self.bindings[name].binding_type = BindingType.NONLOCAL
        else:
            self.bindings[name] = BindingInfo(
                name=name, scope=self, binding_type=BindingType.NONLOCAL
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_binding(self, name: str) -> bool:
        return name in self.bindings

    def is_local(self, name: str) -> bool:
        if name in self.bindings:
            return self.bindings[name].binding_type in (
                BindingType.LOCAL,
                BindingType.PARAMETER,
            )
        return False

    def is_free_variable(self, name: str) -> bool:
        return name in self.free_vars

    def is_cell_variable(self, name: str) -> bool:
        return name in self.cell_vars

    def is_global_variable(self, name: str) -> bool:
        return name in self.global_vars

    def is_nonlocal_variable(self, name: str) -> bool:
        return name in self.nonlocal_vars

    def get_locals(self) -> Dict[str, BindingInfo]:
        return {
            n: b
            for n, b in self.bindings.items()
            if b.binding_type in (BindingType.LOCAL, BindingType.PARAMETER)
        }

    def get_parameters(self) -> List[BindingInfo]:
        return [b for b in self.bindings.values() if b.is_parameter]

    def get_all_names(self) -> Set[str]:
        return set(self.bindings.keys())

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def qualified_name(self) -> str:
        parts: List[str] = []
        scope: Optional[Scope] = self
        while scope is not None:
            parts.append(scope.name)
            scope = scope.parent
        return ".".join(reversed(parts))

    def enclosing_function(self) -> Optional[Scope]:
        scope = self.parent
        while scope is not None:
            if scope.scope_type in (ScopeType.FUNCTION, ScopeType.LAMBDA):
                return scope
            scope = scope.parent
        return None

    def enclosing_class(self) -> Optional[Scope]:
        scope = self.parent
        while scope is not None:
            if scope.scope_type == ScopeType.CLASS:
                return scope
            scope = scope.parent
        return None

    def module_scope(self) -> Scope:
        scope: Scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def ancestors(self) -> List[Scope]:
        result: List[Scope] = []
        scope = self.parent
        while scope is not None:
            result.append(scope)
            scope = scope.parent
        return result

    def descendants(self) -> List[Scope]:
        result: List[Scope] = []
        stack = list(self.children)
        while stack:
            child = stack.pop()
            result.append(child)
            stack.extend(child.children)
        return result

    def __repr__(self) -> str:
        return f"Scope({self.name}, {self.scope_type.name}, bindings={len(self.bindings)})"


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

@dataclass
class ScopeDiagnostic:
    severity: DiagnosticSeverity
    message: str
    location: Optional[SourceLocation] = None
    scope: Optional[Scope] = None
    name: Optional[str] = None

    def __repr__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.name}]{loc}: {self.message}"


# ---------------------------------------------------------------------------
# ScopeTree
# ---------------------------------------------------------------------------

@dataclass
class ScopeTree:
    """Complete tree of scopes for a module."""

    root: Scope
    _node_map: Dict[int, Scope] = field(default_factory=dict)
    diagnostics: List[ScopeDiagnostic] = field(default_factory=list)

    def register_node(self, node: ast.AST, scope: Scope) -> None:
        self._node_map[id(node)] = scope

    def get_scope_for_node(self, node: ast.AST) -> Optional[Scope]:
        return self._node_map.get(id(node))

    def get_binding(self, name: str, scope: Scope) -> Optional[BindingInfo]:
        resolver = LEGBResolver(self)
        result = resolver.resolve(name, scope)
        if result is not None:
            return result[1]
        return None

    def get_all_scopes(self) -> List[Scope]:
        result: List[Scope] = [self.root]
        result.extend(self.root.descendants())
        return result

    def get_scopes_by_type(self, scope_type: ScopeType) -> List[Scope]:
        return [s for s in self.get_all_scopes() if s.scope_type == scope_type]

    def get_function_scopes(self) -> List[Scope]:
        return self.get_scopes_by_type(ScopeType.FUNCTION)

    def get_class_scopes(self) -> List[Scope]:
        return self.get_scopes_by_type(ScopeType.CLASS)

    def find_scope_by_name(self, name: str) -> Optional[Scope]:
        for scope in self.get_all_scopes():
            if scope.name == name:
                return scope
        return None

    def find_scopes_by_name(self, name: str) -> List[Scope]:
        return [s for s in self.get_all_scopes() if s.name == name]

    def add_diagnostic(
        self,
        severity: DiagnosticSeverity,
        message: str,
        location: Optional[SourceLocation] = None,
        scope: Optional[Scope] = None,
        name: Optional[str] = None,
    ) -> None:
        self.diagnostics.append(
            ScopeDiagnostic(
                severity=severity,
                message=message,
                location=location,
                scope=scope,
                name=name,
            )
        )

    def visualize(self) -> str:
        lines: List[str] = []
        self._visualize_scope(self.root, lines, indent=0)
        return "\n".join(lines)

    def _visualize_scope(
        self, scope: Scope, lines: List[str], indent: int
    ) -> None:
        prefix = "  " * indent
        header = f"{prefix}[{scope.scope_type.name}] {scope.name}"
        if scope.is_generator:
            header += " (generator)"
        if scope.is_coroutine:
            header += " (coroutine)"
        lines.append(header)

        if scope.explicit_global_names:
            lines.append(f"{prefix}  globals: {sorted(scope.explicit_global_names)}")
        if scope.explicit_nonlocal_names:
            lines.append(
                f"{prefix}  nonlocals: {sorted(scope.explicit_nonlocal_names)}"
            )

        for name in sorted(scope.bindings):
            bi = scope.bindings[name]
            parts = [f"{prefix}  {name}: {bi.binding_type.name}"]
            if bi.is_parameter:
                parts.append(f"param({bi.parameter_kind.name if bi.parameter_kind else '?'})")
            if bi.is_annotated:
                parts.append("annotated")
            if bi.definition_sites:
                locs = ", ".join(str(d.location) for d in bi.definition_sites)
                parts.append(f"defs=[{locs}]")
            if bi.use_sites:
                locs = ", ".join(str(u.location) for u in bi.use_sites)
                parts.append(f"uses=[{locs}]")
            lines.append(" ".join(parts))

        if scope.free_vars:
            lines.append(f"{prefix}  free_vars: {sorted(scope.free_vars)}")
        if scope.cell_vars:
            lines.append(f"{prefix}  cell_vars: {sorted(scope.cell_vars)}")

        for child in scope.children:
            self._visualize_scope(child, lines, indent + 1)


# ---------------------------------------------------------------------------
# BuiltinScope
# ---------------------------------------------------------------------------

class BuiltinScope:
    """Provides access to Python builtins as a scope-like object."""

    _BUILTIN_FUNCTIONS: FrozenSet[str] = frozenset(
        dir(builtins)
    )

    _BUILTIN_CONSTANTS: FrozenSet[str] = frozenset({
        "True",
        "False",
        "None",
        "NotImplemented",
        "Ellipsis",
        "__debug__",
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
    })

    _BUILTIN_EXCEPTIONS: FrozenSet[str] = frozenset(
        name
        for name in dir(builtins)
        if isinstance(getattr(builtins, name, None), type)
        and issubclass(getattr(builtins, name), BaseException)
    )

    _BUILTIN_TYPES: FrozenSet[str] = frozenset(
        name
        for name in dir(builtins)
        if isinstance(getattr(builtins, name, None), type)
        and not issubclass(getattr(builtins, name), BaseException)
    )

    def __init__(self) -> None:
        self._all_names: FrozenSet[str] = frozenset(dir(builtins))
        self._bindings: Dict[str, BindingInfo] = {}

    def is_builtin(self, name: str) -> bool:
        return name in self._all_names

    def get_binding(self, name: str) -> Optional[BindingInfo]:
        if name not in self._all_names:
            return None
        if name not in self._bindings:
            info = BindingInfo(
                name=name,
                scope=None,
                binding_type=BindingType.BUILTIN,
                is_mutable=False,
            )
            self._bindings[name] = info
        return self._bindings[name]

    def is_builtin_function(self, name: str) -> bool:
        return name in self._BUILTIN_FUNCTIONS and callable(
            getattr(builtins, name, None)
        )

    def is_builtin_type(self, name: str) -> bool:
        return name in self._BUILTIN_TYPES

    def is_builtin_exception(self, name: str) -> bool:
        return name in self._BUILTIN_EXCEPTIONS

    def is_builtin_constant(self, name: str) -> bool:
        return name in self._BUILTIN_CONSTANTS

    def all_names(self) -> FrozenSet[str]:
        return self._all_names


# ---------------------------------------------------------------------------
# LEGBResolver
# ---------------------------------------------------------------------------

class LEGBResolver:
    """Implements the Local-Enclosing-Global-Builtin lookup rule."""

    def __init__(self, scope_tree: ScopeTree) -> None:
        self.scope_tree = scope_tree
        self.builtin_scope = BuiltinScope()

    def resolve(
        self, name: str, scope: Scope
    ) -> Optional[Tuple[Scope, BindingInfo]]:
        # 1. Check explicit global declaration
        if name in scope.explicit_global_names:
            return self._resolve_global(name, scope)

        # 2. Check explicit nonlocal declaration
        if name in scope.explicit_nonlocal_names:
            return self._resolve_nonlocal(name, scope)

        # 3. Local lookup
        local = self._resolve_local(name, scope)
        if local is not None:
            return local

        # 4. Enclosing scopes (skip class scopes per Python semantics)
        enclosing = self._resolve_enclosing(name, scope)
        if enclosing is not None:
            return enclosing

        # 5. Global (module) scope
        module = self._resolve_global(name, scope)
        if module is not None:
            return module

        # 6. Builtin scope
        return self._resolve_builtin(name)

    def _resolve_local(
        self, name: str, scope: Scope
    ) -> Optional[Tuple[Scope, BindingInfo]]:
        info = scope.lookup(name)
        if info is not None and info.binding_type not in (
            BindingType.FREE,
            BindingType.GLOBAL,
            BindingType.NONLOCAL,
        ):
            return (scope, info)
        return None

    def _resolve_enclosing(
        self, name: str, scope: Scope
    ) -> Optional[Tuple[Scope, BindingInfo]]:
        current = scope.parent
        while current is not None:
            # Module scope is handled by _resolve_global
            if current.scope_type == ScopeType.MODULE:
                break

            # Class scopes are skipped in LEGB for nested functions
            if current.scope_type == ScopeType.CLASS:
                current = current.parent
                continue

            # Check for explicit global in enclosing scope – if found,
            # the name should be resolved at module level instead.
            if name in current.explicit_global_names:
                return self._resolve_global(name, scope)

            info = current.lookup(name)
            if info is not None and info.binding_type not in (
                BindingType.FREE,
                BindingType.GLOBAL,
                BindingType.NONLOCAL,
            ):
                return (current, info)

            current = current.parent
        return None

    def _resolve_global(
        self, name: str, scope: Scope
    ) -> Optional[Tuple[Scope, BindingInfo]]:
        module = scope.module_scope()
        info = module.lookup(name)
        if info is not None:
            return (module, info)
        return None

    def _resolve_nonlocal(
        self, name: str, scope: Scope
    ) -> Optional[Tuple[Scope, BindingInfo]]:
        current = scope.parent
        while current is not None:
            if current.scope_type == ScopeType.MODULE:
                break
            if current.scope_type == ScopeType.CLASS:
                current = current.parent
                continue
            info = current.lookup(name)
            if info is not None and info.binding_type not in (
                BindingType.FREE,
                BindingType.GLOBAL,
            ):
                return (current, info)
            current = current.parent
        return None

    def _resolve_builtin(
        self, name: str
    ) -> Optional[Tuple[Scope, BindingInfo]]:
        info = self.builtin_scope.get_binding(name)
        if info is not None:
            # Return with scope=None to indicate builtin
            return (self.scope_tree.root, info)
        return None

    def resolve_all_in_scope(
        self, scope: Scope
    ) -> Dict[str, Tuple[Scope, BindingInfo]]:
        result: Dict[str, Tuple[Scope, BindingInfo]] = {}
        for name in scope.get_all_names():
            resolved = self.resolve(name, scope)
            if resolved is not None:
                result[name] = resolved
        return result


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------

@dataclass
class CapturedVariable:
    name: str
    defining_scope: Scope
    capture_mode: CaptureMode
    is_mutable_capture: bool = False
    is_late_binding_risk: bool = False
    definition_sites: List[DefSite] = field(default_factory=list)
    use_sites: List[UseSite] = field(default_factory=list)


@dataclass
class Closure:
    function_scope: Scope
    captured_vars: List[CapturedVariable] = field(default_factory=list)
    capture_modes: Dict[str, CaptureMode] = field(default_factory=dict)
    is_nested_closure: bool = False
    depth: int = 0

    @property
    def capture_count(self) -> int:
        return len(self.captured_vars)

    @property
    def has_mutable_captures(self) -> bool:
        return any(cv.is_mutable_capture for cv in self.captured_vars)

    @property
    def has_late_binding_risks(self) -> bool:
        return any(cv.is_late_binding_risk for cv in self.captured_vars)


# ---------------------------------------------------------------------------
# ClosureAnalyzer
# ---------------------------------------------------------------------------

class ClosureAnalyzer:
    """Detects closures and analyzes captured variables."""

    def __init__(self, scope_tree: ScopeTree) -> None:
        self.scope_tree = scope_tree
        self.resolver = LEGBResolver(scope_tree)
        self._closures: List[Closure] = []

    def detect_closures(self) -> List[Closure]:
        self._closures = []
        for scope in self.scope_tree.get_all_scopes():
            if scope.scope_type in (ScopeType.FUNCTION, ScopeType.LAMBDA):
                closure = self._analyze_scope_closure(scope)
                if closure is not None:
                    self._closures.append(closure)
        return self._closures

    def _analyze_scope_closure(self, scope: Scope) -> Optional[Closure]:
        captured: List[CapturedVariable] = []
        capture_modes: Dict[str, CaptureMode] = {}

        for name in scope.free_vars:
            resolved = self.resolver.resolve(name, scope)
            if resolved is None:
                continue
            defining_scope, binding_info = resolved
            if defining_scope.scope_type == ScopeType.MODULE:
                continue
            if binding_info.binding_type == BindingType.BUILTIN:
                continue

            is_mutable = self._is_mutable_capture(name, scope, binding_info)
            is_late = self._is_late_binding_risk(name, scope)
            mode = CaptureMode.BY_REFERENCE

            cv = CapturedVariable(
                name=name,
                defining_scope=defining_scope,
                capture_mode=mode,
                is_mutable_capture=is_mutable,
                is_late_binding_risk=is_late,
                definition_sites=list(binding_info.definition_sites),
                use_sites=list(binding_info.use_sites),
            )
            captured.append(cv)
            capture_modes[name] = mode

        if not captured:
            return None

        func_scope = scope.enclosing_function()
        is_nested = func_scope is not None and func_scope.scope_type in (
            ScopeType.FUNCTION,
            ScopeType.LAMBDA,
        )
        depth = 0
        s: Optional[Scope] = scope
        while s is not None:
            if s.scope_type in (ScopeType.FUNCTION, ScopeType.LAMBDA):
                depth += 1
            s = s.parent

        return Closure(
            function_scope=scope,
            captured_vars=captured,
            capture_modes=capture_modes,
            is_nested_closure=is_nested,
            depth=depth,
        )

    def _is_mutable_capture(
        self, name: str, scope: Scope, binding_info: BindingInfo
    ) -> bool:
        if name in scope.explicit_nonlocal_names:
            return True
        for child in scope.descendants():
            if name in child.explicit_nonlocal_names:
                return True
        if len(binding_info.definition_sites) > 1:
            return True
        return False

    def _is_late_binding_risk(self, name: str, scope: Scope) -> bool:
        if scope.scope_type != ScopeType.LAMBDA:
            return False
        parent = scope.parent
        if parent is None:
            return False
        node = parent.node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.AsyncFor)):
                target_names = self._extract_target_names(child.target)
                if name in target_names:
                    for body_node in ast.walk(child):
                        if body_node is scope.node:
                            return True
        return False

    def _extract_target_names(self, target: ast.AST) -> Set[str]:
        names: Set[str] = set()
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names.update(self._extract_target_names(elt))
        elif isinstance(target, ast.Starred):
            names.update(self._extract_target_names(target.value))
        return names

    def compute_cell_and_free_vars(self) -> None:
        all_scopes = self.scope_tree.get_all_scopes()

        for scope in reversed(all_scopes):
            if scope.scope_type == ScopeType.MODULE:
                continue
            for name in list(scope.bindings.keys()):
                bi = scope.bindings[name]
                if bi.binding_type == BindingType.FREE:
                    scope.free_vars.add(name)

        changed = True
        while changed:
            changed = False
            for scope in all_scopes:
                for child in scope.children:
                    for name in child.free_vars:
                        if name in child.explicit_global_names:
                            continue
                        if name in child.explicit_nonlocal_names:
                            continue
                        info = scope.lookup(name)
                        if info is not None and info.binding_type in (
                            BindingType.LOCAL,
                            BindingType.PARAMETER,
                        ):
                            if name not in scope.cell_vars:
                                scope.cell_vars.add(name)
                                info.binding_type = BindingType.CELL
                                changed = True
                        elif scope.scope_type != ScopeType.MODULE:
                            if scope.scope_type == ScopeType.CLASS:
                                pass
                            else:
                                if name not in scope.free_vars:
                                    scope.free_vars.add(name)
                                    changed = True


# ---------------------------------------------------------------------------
# ComprehensionScopeHandler
# ---------------------------------------------------------------------------

class ComprehensionScopeHandler:
    """Handles comprehension-specific scoping rules."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_listcomp(
        self, node: ast.ListComp, parent_scope: Scope
    ) -> Scope:
        return self._handle_comprehension(node, parent_scope, "<listcomp>")

    def handle_setcomp(
        self, node: ast.SetComp, parent_scope: Scope
    ) -> Scope:
        return self._handle_comprehension(node, parent_scope, "<setcomp>")

    def handle_dictcomp(
        self, node: ast.DictComp, parent_scope: Scope
    ) -> Scope:
        return self._handle_comprehension(node, parent_scope, "<dictcomp>")

    def handle_genexp(
        self, node: ast.GeneratorExp, parent_scope: Scope
    ) -> Scope:
        scope = self._handle_comprehension(node, parent_scope, "<genexpr>")
        scope.is_generator = True
        return scope

    def _handle_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
        parent_scope: Scope,
        name: str,
    ) -> Scope:
        comp_scope = Scope(
            name=name,
            scope_type=ScopeType.COMPREHENSION,
            node=node,
            is_comprehension_scope=True,
        )
        parent_scope.add_child(comp_scope)
        self.analyzer.scope_tree.register_node(node, comp_scope)

        generators = node.generators

        # The first iterator is evaluated in the enclosing scope
        if generators:
            first_gen = generators[0]
            self.analyzer._visit_expression(first_gen.iter, parent_scope)

            # The iteration target is defined in the comprehension scope
            self._define_comp_target(first_gen.target, comp_scope)
            comp_scope.bindings[self._target_name(first_gen.target)] = BindingInfo(
                name=self._target_name(first_gen.target),
                scope=comp_scope,
                binding_type=BindingType.LOCAL,
                is_comprehension_iter=True,
            ) if self._target_name(first_gen.target) not in comp_scope.bindings else comp_scope.bindings[self._target_name(first_gen.target)]

            for if_clause in first_gen.ifs:
                self.analyzer._visit_expression(if_clause, comp_scope)

        # Subsequent generators are evaluated in the comprehension scope
        for gen in generators[1:]:
            self.analyzer._visit_expression(gen.iter, comp_scope)
            self._define_comp_target(gen.target, comp_scope)
            for if_clause in gen.ifs:
                self.analyzer._visit_expression(if_clause, comp_scope)

        # Visit the element expression(s)
        if isinstance(node, ast.DictComp):
            self.analyzer._visit_expression(node.key, comp_scope)
            self.analyzer._visit_expression(node.value, comp_scope)
        else:
            self.analyzer._visit_expression(node.elt, comp_scope)

        return comp_scope

    def _define_comp_target(self, target: ast.AST, scope: Scope) -> None:
        if isinstance(target, ast.Name):
            scope.add_binding(
                target.id,
                target,
                BindingType.LOCAL,
                kind="comprehension_target",
                is_comprehension_iter=True,
            )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_comp_target(elt, scope)
        elif isinstance(target, ast.Starred):
            self._define_comp_target(target.value, scope)

    def _target_name(self, target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            names = []
            for elt in target.elts:
                names.append(self._target_name(elt))
            return ",".join(names)
        return "<complex>"

    def detect_iteration_variable_leakage(
        self, scope_tree: ScopeTree
    ) -> List[ScopeDiagnostic]:
        diagnostics: List[ScopeDiagnostic] = []
        for scope in scope_tree.get_all_scopes():
            if scope.scope_type != ScopeType.COMPREHENSION:
                continue
            parent = scope.parent
            if parent is None:
                continue
            for name, bi in scope.bindings.items():
                if bi.is_comprehension_iter and name in parent.bindings:
                    parent_bi = parent.bindings[name]
                    if parent_bi.use_sites:
                        diagnostics.append(
                            ScopeDiagnostic(
                                severity=DiagnosticSeverity.WARNING,
                                message=(
                                    f"Comprehension iteration variable '{name}' "
                                    f"shadows variable in enclosing scope"
                                ),
                                location=SourceLocation.from_node(scope.node),
                                scope=scope,
                                name=name,
                            )
                        )
        return diagnostics


# ---------------------------------------------------------------------------
# ClassScopeHandler
# ---------------------------------------------------------------------------

class ClassScopeHandler:
    """Handles Python class scoping rules."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_class_def(
        self, node: ast.ClassDef, parent_scope: Scope
    ) -> Scope:
        # Class name is bound in the enclosing scope
        parent_scope.add_binding(node.name, node, BindingType.LOCAL, kind="class_def")

        # Decorators are evaluated in the enclosing scope
        for decorator in node.decorator_list:
            self.analyzer._visit_expression(decorator, parent_scope)

        # Base classes are evaluated in the enclosing scope
        for base in node.bases:
            self.analyzer._visit_expression(base, parent_scope)
        for keyword in node.keywords:
            self.analyzer._visit_expression(keyword.value, parent_scope)

        # Create class scope
        class_scope = Scope(
            name=node.name,
            scope_type=ScopeType.CLASS,
            node=node,
        )
        parent_scope.add_child(class_scope)
        self.analyzer.scope_tree.register_node(node, class_scope)

        # __class__ is implicitly a cell variable for classes
        class_scope.add_binding(
            "__class__", node, BindingType.LOCAL, kind="implicit"
        )

        # Visit class body
        for stmt in node.body:
            self.analyzer._visit_statement(stmt, class_scope)

        return class_scope

    def compute_mro(self, class_node: ast.ClassDef, scope: Scope) -> List[str]:
        """C3 linearization for MRO computation."""
        bases = self._get_base_names(class_node)
        if not bases:
            return [class_node.name, "object"]

        base_mros: List[List[str]] = []
        for base_name in bases:
            base_mros.append([base_name, "object"])

        return self._c3_linearize(class_node.name, bases, base_mros)

    def _c3_linearize(
        self,
        class_name: str,
        bases: List[str],
        base_mros: List[List[str]],
    ) -> List[str]:
        result = [class_name]
        sequences = [list(mro) for mro in base_mros] + [list(bases)]
        sequences = [s for s in sequences if s]

        while sequences:
            candidate = None
            for seq in sequences:
                head = seq[0]
                # Check if head appears in the tail of any other sequence
                in_tail = False
                for other_seq in sequences:
                    if head in other_seq[1:]:
                        in_tail = True
                        break
                if not in_tail:
                    candidate = head
                    break

            if candidate is None:
                # Cannot linearize — inconsistent MRO
                remaining = set()
                for seq in sequences:
                    remaining.update(seq)
                result.extend(sorted(remaining))
                break

            result.append(candidate)
            sequences = [
                [item for item in seq if item != candidate] for seq in sequences
            ]
            sequences = [s for s in sequences if s]

        return result

    def _get_base_names(self, node: ast.ClassDef) -> List[str]:
        names: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                names.append(base.id)
            elif isinstance(base, ast.Attribute):
                names.append(ast.dump(base))
            else:
                names.append("<complex>")
        return names

    def resolve_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                if isinstance(keyword.value, ast.Name):
                    return keyword.value.id
                elif isinstance(keyword.value, ast.Attribute):
                    return ast.dump(keyword.value)
        return None

    def get_slots(self, node: ast.ClassDef) -> Optional[List[str]]:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        return self._extract_slot_names(stmt.value)
            elif isinstance(stmt, ast.AnnAssign):
                if (
                    isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "__slots__"
                    and stmt.value is not None
                ):
                    return self._extract_slot_names(stmt.value)
        return None

    def _extract_slot_names(self, value: ast.AST) -> List[str]:
        names: List[str] = []
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.append(elt.value)
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.append(value.value)
        return names

    def classify_attribute(
        self, node: ast.ClassDef, attr_name: str
    ) -> str:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        return "class_variable"
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                    return "class_variable"
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name == "__init__":
                    for init_stmt in ast.walk(stmt):
                        if (
                            isinstance(init_stmt, ast.Assign)
                            and init_stmt.targets
                        ):
                            for t in init_stmt.targets:
                                if (
                                    isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"
                                    and t.attr == attr_name
                                ):
                                    return "instance_variable"
        return "unknown"


# ---------------------------------------------------------------------------
# ImportScopeHandler
# ---------------------------------------------------------------------------

class ImportScopeHandler:
    """Handles import statement effects on scope."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer
        self._imported_modules: Dict[str, List[SourceLocation]] = {}
        self._star_imports: List[Tuple[str, SourceLocation]] = []

    def handle_import(self, node: ast.Import, scope: Scope) -> None:
        for alias in node.names:
            bound_name = alias.asname if alias.asname else alias.name.split(".")[0]
            bi = scope.add_binding(
                bound_name,
                node,
                BindingType.IMPORT,
                kind="import",
                is_imported=True,
                import_module=alias.name,
                import_name=alias.name,
            )
            loc = SourceLocation.from_node(node)
            self._imported_modules.setdefault(alias.name, []).append(loc)

    def handle_import_from(self, node: ast.ImportFrom, scope: Scope) -> None:
        module = node.module or ""
        level = node.level or 0

        for alias in node.names:
            if alias.name == "*":
                self._handle_star_import(node, module, scope)
                continue

            bound_name = alias.asname if alias.asname else alias.name
            scope.add_binding(
                bound_name,
                node,
                BindingType.IMPORT,
                kind="import_from",
                is_imported=True,
                import_module=module,
                import_name=alias.name,
            )

        loc = SourceLocation.from_node(node)
        full_module = ("." * level) + module
        self._imported_modules.setdefault(full_module, []).append(loc)

    def _handle_star_import(
        self, node: ast.ImportFrom, module: str, scope: Scope
    ) -> None:
        loc = SourceLocation.from_node(node)
        self._star_imports.append((module, loc))
        # Star imports bring unknown names; we just mark it
        scope.add_binding(
            f"*:{module}",
            node,
            BindingType.IMPORT,
            kind="star_import",
            is_imported=True,
            import_module=module,
        )

    def detect_circular_imports(
        self, module_imports: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """Find cycles in the import graph using DFS."""
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        path: List[str] = []
        path_set: Set[str] = set()

        def dfs(module: str) -> None:
            if module in path_set:
                cycle_start = path.index(module)
                cycles.append(path[cycle_start:] + [module])
                return
            if module in visited:
                return
            visited.add(module)
            path.append(module)
            path_set.add(module)
            for dep in module_imports.get(module, set()):
                dfs(dep)
            path.pop()
            path_set.discard(module)

        for mod in module_imports:
            dfs(mod)

        return cycles

    def resolve_relative_import(
        self, module: str, level: int, package: Optional[str]
    ) -> Optional[str]:
        if level == 0:
            return module
        if package is None:
            return None
        parts = package.split(".")
        if level > len(parts):
            return None
        base = ".".join(parts[: len(parts) - level + 1])
        if module:
            return f"{base}.{module}"
        return base

    def get_all_imports(self) -> Dict[str, List[SourceLocation]]:
        return dict(self._imported_modules)

    def get_star_imports(self) -> List[Tuple[str, SourceLocation]]:
        return list(self._star_imports)

    def detect_lazy_imports(
        self, scope_tree: ScopeTree
    ) -> List[Tuple[str, Scope, SourceLocation]]:
        results: List[Tuple[str, Scope, SourceLocation]] = []
        for scope in scope_tree.get_all_scopes():
            if scope.scope_type != ScopeType.FUNCTION:
                continue
            for name, bi in scope.bindings.items():
                if bi.is_imported:
                    for ds in bi.definition_sites:
                        results.append((name, scope, ds.location))
        return results


# ---------------------------------------------------------------------------
# DecoratorScopeHandler
# ---------------------------------------------------------------------------

class DecoratorScopeHandler:
    """Handles decorator evaluation and scope effects."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_decorators(
        self,
        decorator_list: List[ast.AST],
        scope: Scope,
    ) -> List[str]:
        decorator_names: List[str] = []
        for dec in decorator_list:
            name = self._decorator_name(dec)
            decorator_names.append(name)
            self.analyzer._visit_expression(dec, scope)
        return decorator_names

    def _decorator_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._decorator_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return "<complex>"

    def is_staticmethod(self, decorators: List[str]) -> bool:
        return "staticmethod" in decorators

    def is_classmethod(self, decorators: List[str]) -> bool:
        return "classmethod" in decorators

    def is_property(self, decorators: List[str]) -> bool:
        return "property" in decorators or any(
            d.endswith(".setter") or d.endswith(".getter") or d.endswith(".deleter")
            for d in decorators
        )

    def get_effect_on_scope(
        self, decorators: List[str]
    ) -> Dict[str, Any]:
        effects: Dict[str, Any] = {
            "is_static": self.is_staticmethod(decorators),
            "is_classmethod": self.is_classmethod(decorators),
            "is_property": self.is_property(decorators),
            "removes_first_param": self.is_staticmethod(decorators),
            "changes_first_param": self.is_classmethod(decorators),
        }
        return effects


# ---------------------------------------------------------------------------
# ExceptionScopeHandler
# ---------------------------------------------------------------------------

class ExceptionScopeHandler:
    """Handles exception variable scoping (Python 3: exception var deleted after block)."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_except_handler(
        self, node: ast.ExceptHandler, scope: Scope
    ) -> Optional[str]:
        if node.type is not None:
            self.analyzer._visit_expression(node.type, scope)

        exception_var_name: Optional[str] = None
        if node.name is not None:
            exception_var_name = node.name
            bi = scope.add_binding(
                node.name,
                node,
                BindingType.LOCAL,
                kind="exception_var",
                is_exception_var=True,
            )

        for stmt in node.body:
            self.analyzer._visit_statement(stmt, scope)

        # In Python 3, the exception variable is implicitly deleted after
        # the except block. We record this.
        if exception_var_name is not None and exception_var_name in scope.bindings:
            bi = scope.bindings[exception_var_name]
            bi.is_exception_var = True

        return exception_var_name

    def detect_exception_var_use_after_block(
        self, scope_tree: ScopeTree
    ) -> List[ScopeDiagnostic]:
        diagnostics: List[ScopeDiagnostic] = []
        for scope in scope_tree.get_all_scopes():
            for name, bi in scope.bindings.items():
                if not bi.is_exception_var:
                    continue
                # Check if there are uses after the except block definition
                if not bi.definition_sites:
                    continue
                def_loc = bi.definition_sites[-1].location
                for use in bi.use_sites:
                    if use.location.lineno > def_loc.lineno + _count_except_body_lines(bi):
                        diagnostics.append(
                            ScopeDiagnostic(
                                severity=DiagnosticSeverity.ERROR,
                                message=(
                                    f"Exception variable '{name}' used after "
                                    f"except block (deleted in Python 3)"
                                ),
                                location=use.location,
                                scope=scope,
                                name=name,
                            )
                        )
        return diagnostics


def _count_except_body_lines(bi: BindingInfo) -> int:
    """Heuristic: count lines in the except handler body."""
    if not bi.definition_sites:
        return 0
    node = bi.definition_sites[-1].node
    if isinstance(node, ast.ExceptHandler) and node.body:
        first_line = node.body[0].lineno if hasattr(node.body[0], "lineno") else 0
        last_line = node.body[-1].end_lineno if hasattr(node.body[-1], "end_lineno") and node.body[-1].end_lineno else 0
        if first_line and last_line:
            return last_line - first_line + 1
    return 5  # conservative fallback


# ---------------------------------------------------------------------------
# WalrusOperatorHandler
# ---------------------------------------------------------------------------

class WalrusOperatorHandler:
    """Handles := (walrus operator / named expression) scoping."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_named_expr(
        self, node: ast.NamedExpr, current_scope: Scope
    ) -> Scope:
        """
        In comprehension scope, := leaks to the enclosing *function* or *module* scope.
        Outside comprehensions, it binds in the current scope.
        """
        target_scope = self._find_target_scope(current_scope)
        name = node.target.id

        target_scope.add_binding(
            name,
            node,
            BindingType.LOCAL,
            kind="walrus",
            is_walrus=True,
        )
        if target_scope is not current_scope:
            current_scope.add_use(name, node.target, context="load")

        self.analyzer._visit_expression(node.value, current_scope)
        return target_scope

    def _find_target_scope(self, scope: Scope) -> Scope:
        current = scope
        while current.scope_type == ScopeType.COMPREHENSION:
            if current.parent is None:
                break
            current = current.parent
        return current

    def detect_walrus_in_comprehension(
        self, scope_tree: ScopeTree
    ) -> List[ScopeDiagnostic]:
        diagnostics: List[ScopeDiagnostic] = []
        for scope in scope_tree.get_all_scopes():
            if scope.scope_type != ScopeType.COMPREHENSION:
                continue
            for name, bi in scope.bindings.items():
                if bi.is_walrus:
                    diagnostics.append(
                        ScopeDiagnostic(
                            severity=DiagnosticSeverity.INFO,
                            message=(
                                f"Walrus operator variable '{name}' in comprehension "
                                f"leaks to enclosing scope"
                            ),
                            location=(
                                bi.definition_sites[0].location
                                if bi.definition_sites
                                else None
                            ),
                            scope=scope,
                            name=name,
                        )
                    )
        return diagnostics


# ---------------------------------------------------------------------------
# MatchStatementHandler
# ---------------------------------------------------------------------------

class MatchStatementHandler:
    """Handles match/case pattern variable binding (Python 3.10+)."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_match(self, node: ast.Match, scope: Scope) -> None:
        self.analyzer._visit_expression(node.subject, scope)
        for case in node.cases:
            self._handle_case(case, scope)

    def _handle_case(self, case: ast.match_case, scope: Scope) -> None:
        self._visit_pattern(case.pattern, scope)
        if case.guard is not None:
            self.analyzer._visit_expression(case.guard, scope)
        for stmt in case.body:
            self.analyzer._visit_statement(stmt, scope)

    def _visit_pattern(self, pattern: ast.AST, scope: Scope) -> None:
        if isinstance(pattern, ast.MatchValue):
            self.analyzer._visit_expression(pattern.value, scope)

        elif isinstance(pattern, ast.MatchSingleton):
            pass  # True, False, None — no binding

        elif isinstance(pattern, ast.MatchSequence):
            for p in pattern.patterns:
                self._visit_pattern(p, scope)

        elif isinstance(pattern, ast.MatchMapping):
            for key in pattern.keys:
                self.analyzer._visit_expression(key, scope)
            for p in pattern.patterns:
                self._visit_pattern(p, scope)
            if pattern.rest is not None:
                scope.add_binding(
                    pattern.rest,
                    pattern,
                    BindingType.LOCAL,
                    kind="match_rest",
                    is_match_var=True,
                )

        elif isinstance(pattern, ast.MatchClass):
            self.analyzer._visit_expression(pattern.cls, scope)
            for p in pattern.patterns:
                self._visit_pattern(p, scope)
            for p in pattern.kwd_patterns:
                self._visit_pattern(p, scope)

        elif isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                scope.add_binding(
                    pattern.name,
                    pattern,
                    BindingType.LOCAL,
                    kind="match_star",
                    is_match_var=True,
                )

        elif isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._visit_pattern(pattern.pattern, scope)
            if pattern.name is not None:
                scope.add_binding(
                    pattern.name,
                    pattern,
                    BindingType.LOCAL,
                    kind="match_as",
                    is_match_var=True,
                )

        elif isinstance(pattern, ast.MatchOr):
            for p in pattern.patterns:
                self._visit_pattern(p, scope)


# ---------------------------------------------------------------------------
# TypeParamHandler  (PEP 695 – Python 3.12+)
# ---------------------------------------------------------------------------

class TypeParamHandler:
    """Handles PEP 695 type parameter scopes."""

    def __init__(self, analyzer: ScopeAnalyzer) -> None:
        self.analyzer = analyzer

    def handle_type_alias(self, node: ast.AST, scope: Scope) -> Scope:
        """Handle: type X[T] = ..."""
        type_scope = Scope(
            name="<type_params>",
            scope_type=ScopeType.TYPE_PARAM,
            node=node,
        )
        scope.add_child(type_scope)
        self.analyzer.scope_tree.register_node(node, type_scope)

        # Extract type_params if available (Python 3.12+)
        type_params = getattr(node, "type_params", [])
        for tp in type_params:
            self._visit_type_param(tp, type_scope)

        # The name is bound in the enclosing scope
        name = getattr(node, "name", None)
        if name and isinstance(name, ast.Name):
            scope.add_binding(name.id, node, BindingType.LOCAL, kind="type_alias")
        elif name and isinstance(name, str):
            scope.add_binding(name, node, BindingType.LOCAL, kind="type_alias")

        # Visit the value in the type param scope
        value = getattr(node, "value", None)
        if value is not None:
            self.analyzer._visit_expression(value, type_scope)

        return type_scope

    def handle_generic_function(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], scope: Scope
    ) -> Optional[Scope]:
        type_params = getattr(node, "type_params", [])
        if not type_params:
            return None

        type_scope = Scope(
            name=f"<type_params:{node.name}>",
            scope_type=ScopeType.TYPE_PARAM,
            node=node,
        )
        scope.add_child(type_scope)

        for tp in type_params:
            self._visit_type_param(tp, type_scope)

        return type_scope

    def handle_generic_class(
        self, node: ast.ClassDef, scope: Scope
    ) -> Optional[Scope]:
        type_params = getattr(node, "type_params", [])
        if not type_params:
            return None

        type_scope = Scope(
            name=f"<type_params:{node.name}>",
            scope_type=ScopeType.TYPE_PARAM,
            node=node,
        )
        scope.add_child(type_scope)

        for tp in type_params:
            self._visit_type_param(tp, type_scope)

        return type_scope

    def _visit_type_param(self, tp: ast.AST, scope: Scope) -> None:
        # ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple in 3.12
        name = getattr(tp, "name", None)
        if name:
            scope.add_binding(
                name,
                tp,
                BindingType.LOCAL,
                kind="type_param",
                is_type_param=True,
            )
        bound = getattr(tp, "bound", None)
        if bound is not None:
            self.analyzer._visit_expression(bound, scope)


# ---------------------------------------------------------------------------
# ScopeValidator
# ---------------------------------------------------------------------------

class ScopeValidator:
    """Validates the results of scope analysis."""

    def __init__(self, scope_tree: ScopeTree) -> None:
        self.scope_tree = scope_tree
        self.resolver = LEGBResolver(scope_tree)
        self.builtin_scope = BuiltinScope()
        self._diagnostics: List[ScopeDiagnostic] = []

    def validate(self) -> List[ScopeDiagnostic]:
        self._diagnostics = []
        for scope in self.scope_tree.get_all_scopes():
            self._validate_scope(scope)
        return self._diagnostics

    def _validate_scope(self, scope: Scope) -> None:
        self._check_undefined_names(scope)
        self._check_global_declarations(scope)
        self._check_nonlocal_declarations(scope)
        self._check_conflicting_bindings(scope)
        self._check_unused_variables(scope)
        self._check_unused_imports(scope)
        self._check_shadowing(scope)

    def _check_undefined_names(self, scope: Scope) -> None:
        for name, bi in scope.bindings.items():
            if name.startswith("*:"):
                continue  # star import marker
            if not bi.is_defined and bi.use_sites:
                if bi.binding_type == BindingType.FREE:
                    resolved = self.resolver.resolve(name, scope)
                    if resolved is None and not self.builtin_scope.is_builtin(name):
                        self._diagnostics.append(
                            ScopeDiagnostic(
                                severity=DiagnosticSeverity.ERROR,
                                message=f"Undefined name '{name}'",
                                location=(
                                    bi.use_sites[0].location if bi.use_sites else None
                                ),
                                scope=scope,
                                name=name,
                            )
                        )

    def _check_global_declarations(self, scope: Scope) -> None:
        if scope.scope_type == ScopeType.MODULE:
            if scope.explicit_global_names:
                for name in scope.explicit_global_names:
                    self._diagnostics.append(
                        ScopeDiagnostic(
                            severity=DiagnosticSeverity.WARNING,
                            message=f"'global' declaration of '{name}' at module level is redundant",
                            scope=scope,
                            name=name,
                        )
                    )
            return

        for name in scope.explicit_global_names:
            if name in scope.explicit_nonlocal_names:
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Name '{name}' is both global and nonlocal",
                        scope=scope,
                        name=name,
                    )
                )
            bi = scope.lookup(name)
            if bi is not None and bi.is_parameter:
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Name '{name}' is parameter and global",
                        scope=scope,
                        name=name,
                    )
                )

    def _check_nonlocal_declarations(self, scope: Scope) -> None:
        if scope.scope_type == ScopeType.MODULE:
            for name in scope.explicit_nonlocal_names:
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"'nonlocal' declaration at module level for '{name}'",
                        scope=scope,
                        name=name,
                    )
                )
            return

        for name in scope.explicit_nonlocal_names:
            resolved = self.resolver._resolve_nonlocal(name, scope)
            if resolved is None:
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"No binding for nonlocal '{name}' found in enclosing scopes",
                        scope=scope,
                        name=name,
                    )
                )
            bi = scope.lookup(name)
            if bi is not None and bi.is_parameter:
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Name '{name}' is parameter and nonlocal",
                        scope=scope,
                        name=name,
                    )
                )

    def _check_conflicting_bindings(self, scope: Scope) -> None:
        for name in scope.explicit_global_names:
            if name in scope.explicit_nonlocal_names:
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Name '{name}' is declared both global and nonlocal in {scope.name}",
                        scope=scope,
                        name=name,
                    )
                )

    def _check_unused_variables(self, scope: Scope) -> None:
        if scope.scope_type == ScopeType.MODULE:
            return
        for name, bi in scope.bindings.items():
            if name.startswith("_"):
                continue
            if bi.is_parameter and name == "self":
                continue
            if bi.is_parameter and name == "cls":
                continue
            if bi.is_imported:
                continue
            if bi.is_defined and not bi.is_used and bi.binding_type in (
                BindingType.LOCAL,
                BindingType.PARAMETER,
            ):
                # Check if it's used as a cell variable by children
                if name in scope.cell_vars:
                    continue
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        message=f"Variable '{name}' is defined but never used",
                        location=(
                            bi.definition_sites[0].location
                            if bi.definition_sites
                            else None
                        ),
                        scope=scope,
                        name=name,
                    )
                )

    def _check_unused_imports(self, scope: Scope) -> None:
        for name, bi in scope.bindings.items():
            if name.startswith("*:"):
                continue
            if bi.is_imported and not bi.is_used and not name.startswith("_"):
                self._diagnostics.append(
                    ScopeDiagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        message=f"Imported name '{name}' is unused",
                        location=(
                            bi.definition_sites[0].location
                            if bi.definition_sites
                            else None
                        ),
                        scope=scope,
                        name=name,
                    )
                )

    def _check_shadowing(self, scope: Scope) -> None:
        if scope.scope_type == ScopeType.MODULE:
            return
        for name, bi in scope.bindings.items():
            if not bi.is_defined:
                continue
            if name.startswith("_"):
                continue
            if bi.binding_type in (BindingType.GLOBAL, BindingType.NONLOCAL):
                continue
            parent = scope.parent
            while parent is not None:
                if parent.scope_type == ScopeType.MODULE:
                    break
                if parent.scope_type == ScopeType.CLASS:
                    parent = parent.parent
                    continue
                if name in parent.bindings:
                    parent_bi = parent.bindings[name]
                    if parent_bi.is_defined and parent_bi.binding_type in (
                        BindingType.LOCAL,
                        BindingType.PARAMETER,
                    ):
                        self._diagnostics.append(
                            ScopeDiagnostic(
                                severity=DiagnosticSeverity.INFO,
                                message=(
                                    f"Variable '{name}' shadows variable in "
                                    f"enclosing scope '{parent.name}'"
                                ),
                                location=(
                                    bi.definition_sites[0].location
                                    if bi.definition_sites
                                    else None
                                ),
                                scope=scope,
                                name=name,
                            )
                        )
                        break
                parent = parent.parent


# ---------------------------------------------------------------------------
# ScopeStatistics
# ---------------------------------------------------------------------------

@dataclass
class ScopeStatistics:
    """Collects statistics about a scope tree."""

    total_scopes: int = 0
    scope_type_counts: Dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    total_bindings: int = 0
    total_free_vars: int = 0
    total_cell_vars: int = 0
    closure_count: int = 0
    generator_count: int = 0
    coroutine_count: int = 0
    comprehension_count: int = 0
    class_count: int = 0
    function_count: int = 0
    bindings_per_scope: Dict[str, int] = field(default_factory=dict)
    deepest_scope_name: str = ""
    widest_scope_name: str = ""
    widest_scope_bindings: int = 0

    @classmethod
    def from_scope_tree(cls, scope_tree: ScopeTree) -> ScopeStatistics:
        stats = cls()
        all_scopes = scope_tree.get_all_scopes()
        stats.total_scopes = len(all_scopes)

        for scope in all_scopes:
            type_name = scope.scope_type.name
            stats.scope_type_counts[type_name] = (
                stats.scope_type_counts.get(type_name, 0) + 1
            )

            if scope.depth > stats.max_depth:
                stats.max_depth = scope.depth
                stats.deepest_scope_name = scope.qualified_name

            num_bindings = len(scope.bindings)
            stats.total_bindings += num_bindings
            stats.bindings_per_scope[scope.qualified_name] = num_bindings

            if num_bindings > stats.widest_scope_bindings:
                stats.widest_scope_bindings = num_bindings
                stats.widest_scope_name = scope.qualified_name

            stats.total_free_vars += len(scope.free_vars)
            stats.total_cell_vars += len(scope.cell_vars)

            if scope.is_generator:
                stats.generator_count += 1
            if scope.is_coroutine:
                stats.coroutine_count += 1
            if scope.scope_type == ScopeType.COMPREHENSION:
                stats.comprehension_count += 1
            if scope.scope_type == ScopeType.CLASS:
                stats.class_count += 1
            if scope.scope_type == ScopeType.FUNCTION:
                stats.function_count += 1

        # Count closures
        analyzer = ClosureAnalyzer(scope_tree)
        closures = analyzer.detect_closures()
        stats.closure_count = len(closures)

        return stats

    def summary(self) -> str:
        lines = [
            f"Scope Statistics:",
            f"  Total scopes: {self.total_scopes}",
            f"  Max depth: {self.max_depth} ({self.deepest_scope_name})",
            f"  Total bindings: {self.total_bindings}",
            f"  Widest scope: {self.widest_scope_name} ({self.widest_scope_bindings} bindings)",
            f"  Free variables: {self.total_free_vars}",
            f"  Cell variables: {self.total_cell_vars}",
            f"  Closures: {self.closure_count}",
            f"  Functions: {self.function_count}",
            f"  Classes: {self.class_count}",
            f"  Generators: {self.generator_count}",
            f"  Coroutines: {self.coroutine_count}",
            f"  Comprehensions: {self.comprehension_count}",
            f"  Scope type distribution:",
        ]
        for type_name, count in sorted(self.scope_type_counts.items()):
            lines.append(f"    {type_name}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ScopeAnalyzer  — main driver
# ---------------------------------------------------------------------------

class ScopeAnalyzer:
    """
    Main scope analysis driver.  Walks a Python AST and builds a complete
    ScopeTree that records every binding, use, free variable, cell variable,
    global declaration, nonlocal declaration, closure, etc.
    """

    def __init__(self) -> None:
        self.scope_tree: ScopeTree = None  # type: ignore[assignment]
        self.builtin_scope = BuiltinScope()

        # Sub-handlers (initialised in analyze())
        self._comp_handler: ComprehensionScopeHandler = None  # type: ignore[assignment]
        self._class_handler: ClassScopeHandler = None  # type: ignore[assignment]
        self._import_handler: ImportScopeHandler = None  # type: ignore[assignment]
        self._decorator_handler: DecoratorScopeHandler = None  # type: ignore[assignment]
        self._exception_handler: ExceptionScopeHandler = None  # type: ignore[assignment]
        self._walrus_handler: WalrusOperatorHandler = None  # type: ignore[assignment]
        self._match_handler: MatchStatementHandler = None  # type: ignore[assignment]
        self._type_param_handler: TypeParamHandler = None  # type: ignore[assignment]

        self._current_scope: Optional[Scope] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, module_ast: ast.Module) -> ScopeTree:
        """Analyze a module AST and return a complete ScopeTree."""
        root = Scope(
            name="<module>",
            scope_type=ScopeType.MODULE,
            node=module_ast,
        )
        self.scope_tree = ScopeTree(root=root)
        self.scope_tree.register_node(module_ast, root)

        self._comp_handler = ComprehensionScopeHandler(self)
        self._class_handler = ClassScopeHandler(self)
        self._import_handler = ImportScopeHandler(self)
        self._decorator_handler = DecoratorScopeHandler(self)
        self._exception_handler = ExceptionScopeHandler(self)
        self._walrus_handler = WalrusOperatorHandler(self)
        self._match_handler = MatchStatementHandler(self)
        self._type_param_handler = TypeParamHandler(self)

        self._current_scope = root

        # Pass 1: Walk all statements and build scope tree + bindings
        for stmt in module_ast.body:
            self._visit_statement(stmt, root)

        # Pass 2: Compute free / cell variables
        closure_analyzer = ClosureAnalyzer(self.scope_tree)
        closure_analyzer.compute_cell_and_free_vars()

        return self.scope_tree

    # ------------------------------------------------------------------
    # Statement visitors
    # ------------------------------------------------------------------

    def _visit_statement(self, node: ast.AST, scope: Scope) -> None:
        self.scope_tree.register_node(node, scope)
        self._current_scope = scope

        if isinstance(node, ast.FunctionDef):
            self._visit_function_def(node, scope)
        elif isinstance(node, ast.AsyncFunctionDef):
            self._visit_async_function_def(node, scope)
        elif isinstance(node, ast.ClassDef):
            self._visit_class_def(node, scope)
        elif isinstance(node, ast.Return):
            self._visit_return(node, scope)
        elif isinstance(node, ast.Delete):
            self._visit_delete(node, scope)
        elif isinstance(node, ast.Assign):
            self._visit_assign(node, scope)
        elif isinstance(node, ast.AugAssign):
            self._visit_aug_assign(node, scope)
        elif isinstance(node, ast.AnnAssign):
            self._visit_ann_assign(node, scope)
        elif isinstance(node, ast.For):
            self._visit_for(node, scope)
        elif isinstance(node, ast.AsyncFor):
            self._visit_async_for(node, scope)
        elif isinstance(node, ast.While):
            self._visit_while(node, scope)
        elif isinstance(node, ast.If):
            self._visit_if(node, scope)
        elif isinstance(node, ast.With):
            self._visit_with(node, scope)
        elif isinstance(node, ast.AsyncWith):
            self._visit_async_with(node, scope)
        elif isinstance(node, ast.Raise):
            self._visit_raise(node, scope)
        elif isinstance(node, ast.Try):
            self._visit_try(node, scope)
        elif isinstance(node, (ast.Import,)):
            self._import_handler.handle_import(node, scope)
        elif isinstance(node, ast.ImportFrom):
            self._import_handler.handle_import_from(node, scope)
        elif isinstance(node, ast.Global):
            self._visit_global(node, scope)
        elif isinstance(node, ast.Nonlocal):
            self._visit_nonlocal(node, scope)
        elif isinstance(node, ast.Expr):
            self._visit_expression(node.value, scope)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Break):
            pass
        elif isinstance(node, ast.Continue):
            pass
        elif isinstance(node, ast.Assert):
            self._visit_assert(node, scope)
        elif isinstance(node, ast.Match):
            self._match_handler.handle_match(node, scope)
        elif hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):  # type: ignore[attr-defined]
            self._visit_try_star(node, scope)
        elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):  # type: ignore[attr-defined]
            self._type_param_handler.handle_type_alias(node, scope)
        else:
            # Fallback: walk children generically
            self._generic_visit_stmt(node, scope)

    def _visit_function_def(
        self, node: ast.FunctionDef, scope: Scope
    ) -> None:
        # Decorators evaluated in enclosing scope
        decorator_names = self._decorator_handler.handle_decorators(
            node.decorator_list, scope
        )

        # Function name bound in enclosing scope
        scope.add_binding(node.name, node, BindingType.LOCAL, kind="function_def")

        # Handle PEP 695 type params
        type_scope = self._type_param_handler.handle_generic_function(node, scope)
        param_parent = type_scope if type_scope is not None else scope

        # Default values evaluated in enclosing scope
        self._visit_defaults(node.args, scope)

        # Return annotation in enclosing scope
        if node.returns:
            self._visit_expression(node.returns, scope)

        # Create function scope
        func_scope = Scope(
            name=node.name,
            scope_type=ScopeType.FUNCTION,
            node=node,
        )
        param_parent.add_child(func_scope)
        self.scope_tree.register_node(node, func_scope)

        # Is it a generator?
        func_scope.is_generator = self._contains_yield(node)

        # Bind parameters
        self._bind_parameters(node.args, func_scope)

        # Visit function body
        for stmt in node.body:
            self._visit_statement(stmt, func_scope)

    def _visit_async_function_def(
        self, node: ast.AsyncFunctionDef, scope: Scope
    ) -> None:
        decorator_names = self._decorator_handler.handle_decorators(
            node.decorator_list, scope
        )

        scope.add_binding(node.name, node, BindingType.LOCAL, kind="function_def")

        type_scope = self._type_param_handler.handle_generic_function(node, scope)
        param_parent = type_scope if type_scope is not None else scope

        self._visit_defaults(node.args, scope)
        if node.returns:
            self._visit_expression(node.returns, scope)

        func_scope = Scope(
            name=node.name,
            scope_type=ScopeType.FUNCTION,
            node=node,
            is_coroutine=True,
        )
        param_parent.add_child(func_scope)
        self.scope_tree.register_node(node, func_scope)

        func_scope.is_generator = self._contains_yield(node)

        self._bind_parameters(node.args, func_scope)

        for stmt in node.body:
            self._visit_statement(stmt, func_scope)

    def _visit_class_def(self, node: ast.ClassDef, scope: Scope) -> None:
        self._class_handler.handle_class_def(node, scope)

    def _visit_return(self, node: ast.Return, scope: Scope) -> None:
        if node.value is not None:
            self._visit_expression(node.value, scope)

    def _visit_delete(self, node: ast.Delete, scope: Scope) -> None:
        for target in node.targets:
            self._visit_target(target, scope, context="del")

    def _visit_assign(self, node: ast.Assign, scope: Scope) -> None:
        self._visit_expression(node.value, scope)
        for target in node.targets:
            self._visit_target(target, scope, context="store")

    def _visit_aug_assign(self, node: ast.AugAssign, scope: Scope) -> None:
        self._visit_expression(node.value, scope)
        # AugAssign both reads and writes
        self._visit_target(node.target, scope, context="store")
        if isinstance(node.target, ast.Name):
            scope.add_use(node.target.id, node.target, context="load")

    def _visit_ann_assign(self, node: ast.AnnAssign, scope: Scope) -> None:
        self._visit_expression(node.annotation, scope)
        if node.value is not None:
            self._visit_expression(node.value, scope)
        if node.target is not None:
            if isinstance(node.target, ast.Name):
                bi = scope.add_binding(
                    node.target.id,
                    node,
                    BindingType.LOCAL,
                    kind="annotation",
                    is_annotated=True,
                    annotation=node.annotation,
                )
                if node.value is None:
                    # Annotation without value — mark as annotated but not defined
                    bi.is_annotated = True
            else:
                self._visit_target(node.target, scope, context="store")

    def _visit_for(self, node: ast.For, scope: Scope) -> None:
        self._visit_expression(node.iter, scope)
        self._visit_target(node.target, scope, context="store")
        for stmt in node.body:
            self._visit_statement(stmt, scope)
        for stmt in node.orelse:
            self._visit_statement(stmt, scope)

    def _visit_async_for(self, node: ast.AsyncFor, scope: Scope) -> None:
        self._visit_expression(node.iter, scope)
        self._visit_target(node.target, scope, context="store")
        for stmt in node.body:
            self._visit_statement(stmt, scope)
        for stmt in node.orelse:
            self._visit_statement(stmt, scope)

    def _visit_while(self, node: ast.While, scope: Scope) -> None:
        self._visit_expression(node.test, scope)
        for stmt in node.body:
            self._visit_statement(stmt, scope)
        for stmt in node.orelse:
            self._visit_statement(stmt, scope)

    def _visit_if(self, node: ast.If, scope: Scope) -> None:
        self._visit_expression(node.test, scope)
        for stmt in node.body:
            self._visit_statement(stmt, scope)
        for stmt in node.orelse:
            self._visit_statement(stmt, scope)

    def _visit_with(self, node: ast.With, scope: Scope) -> None:
        for item in node.items:
            self._visit_expression(item.context_expr, scope)
            if item.optional_vars is not None:
                self._visit_target(item.optional_vars, scope, context="store")
        for stmt in node.body:
            self._visit_statement(stmt, scope)

    def _visit_async_with(self, node: ast.AsyncWith, scope: Scope) -> None:
        for item in node.items:
            self._visit_expression(item.context_expr, scope)
            if item.optional_vars is not None:
                self._visit_target(item.optional_vars, scope, context="store")
        for stmt in node.body:
            self._visit_statement(stmt, scope)

    def _visit_raise(self, node: ast.Raise, scope: Scope) -> None:
        if node.exc is not None:
            self._visit_expression(node.exc, scope)
        if node.cause is not None:
            self._visit_expression(node.cause, scope)

    def _visit_try(self, node: ast.Try, scope: Scope) -> None:
        for stmt in node.body:
            self._visit_statement(stmt, scope)
        for handler in node.handlers:
            self._exception_handler.handle_except_handler(handler, scope)
        for stmt in node.orelse:
            self._visit_statement(stmt, scope)
        for stmt in node.finalbody:
            self._visit_statement(stmt, scope)

    def _visit_try_star(self, node: ast.AST, scope: Scope) -> None:
        """Handle try/except* (Python 3.11+)."""
        body = getattr(node, "body", [])
        handlers = getattr(node, "handlers", [])
        orelse = getattr(node, "orelse", [])
        finalbody = getattr(node, "finalbody", [])

        for stmt in body:
            self._visit_statement(stmt, scope)
        for handler in handlers:
            self._exception_handler.handle_except_handler(handler, scope)
        for stmt in orelse:
            self._visit_statement(stmt, scope)
        for stmt in finalbody:
            self._visit_statement(stmt, scope)

    def _visit_global(self, node: ast.Global, scope: Scope) -> None:
        for name in node.names:
            scope.add_global(name)

    def _visit_nonlocal(self, node: ast.Nonlocal, scope: Scope) -> None:
        for name in node.names:
            scope.add_nonlocal(name)

    def _visit_assert(self, node: ast.Assert, scope: Scope) -> None:
        self._visit_expression(node.test, scope)
        if node.msg is not None:
            self._visit_expression(node.msg, scope)

    def _generic_visit_stmt(self, node: ast.AST, scope: Scope) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._visit_expression(child, scope)
            elif isinstance(child, ast.stmt):
                self._visit_statement(child, scope)

    # ------------------------------------------------------------------
    # Expression visitors
    # ------------------------------------------------------------------

    def _visit_expression(self, node: ast.AST, scope: Scope) -> None:
        if node is None:
            return
        self.scope_tree.register_node(node, scope)

        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._visit_expression(value, scope)

        elif isinstance(node, ast.NamedExpr):
            self._walrus_handler.handle_named_expr(node, scope)

        elif isinstance(node, ast.BinOp):
            self._visit_expression(node.left, scope)
            self._visit_expression(node.right, scope)

        elif isinstance(node, ast.UnaryOp):
            self._visit_expression(node.operand, scope)

        elif isinstance(node, ast.Lambda):
            self._visit_lambda(node, scope)

        elif isinstance(node, ast.IfExp):
            self._visit_expression(node.test, scope)
            self._visit_expression(node.body, scope)
            self._visit_expression(node.orelse, scope)

        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key is not None:
                    self._visit_expression(key, scope)
            for value in node.values:
                self._visit_expression(value, scope)

        elif isinstance(node, ast.Set):
            for elt in node.elts:
                self._visit_expression(elt, scope)

        elif isinstance(node, ast.ListComp):
            self._comp_handler.handle_listcomp(node, scope)

        elif isinstance(node, ast.SetComp):
            self._comp_handler.handle_setcomp(node, scope)

        elif isinstance(node, ast.DictComp):
            self._comp_handler.handle_dictcomp(node, scope)

        elif isinstance(node, ast.GeneratorExp):
            self._comp_handler.handle_genexp(node, scope)

        elif isinstance(node, ast.Await):
            self._visit_expression(node.value, scope)

        elif isinstance(node, ast.Yield):
            if node.value is not None:
                self._visit_expression(node.value, scope)

        elif isinstance(node, ast.YieldFrom):
            self._visit_expression(node.value, scope)

        elif isinstance(node, ast.Compare):
            self._visit_expression(node.left, scope)
            for comp in node.comparators:
                self._visit_expression(comp, scope)

        elif isinstance(node, ast.Call):
            self._visit_expression(node.func, scope)
            for arg in node.args:
                self._visit_expression(arg, scope)
            for keyword in node.keywords:
                self._visit_expression(keyword.value, scope)

        elif isinstance(node, ast.FormattedValue):
            self._visit_expression(node.value, scope)
            if node.format_spec is not None:
                self._visit_expression(node.format_spec, scope)

        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                self._visit_expression(value, scope)

        elif isinstance(node, ast.Constant):
            pass  # No scope effect

        elif isinstance(node, ast.Attribute):
            self._visit_expression(node.value, scope)

        elif isinstance(node, ast.Subscript):
            self._visit_expression(node.value, scope)
            self._visit_expression(node.slice, scope)

        elif isinstance(node, ast.Starred):
            self._visit_expression(node.value, scope)

        elif isinstance(node, ast.Name):
            self._visit_name(node, scope)

        elif isinstance(node, ast.List):
            for elt in node.elts:
                self._visit_expression(elt, scope)

        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                self._visit_expression(elt, scope)

        elif isinstance(node, ast.Slice):
            if node.lower is not None:
                self._visit_expression(node.lower, scope)
            if node.upper is not None:
                self._visit_expression(node.upper, scope)
            if node.step is not None:
                self._visit_expression(node.step, scope)

        else:
            # Generic fallback for unknown expression types
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._visit_expression(child, scope)

    def _visit_name(self, node: ast.Name, scope: Scope) -> None:
        name = node.id
        ctx = node.ctx

        if isinstance(ctx, ast.Store):
            if name not in scope.explicit_global_names and name not in scope.explicit_nonlocal_names:
                if name not in scope.bindings or scope.bindings[name].binding_type == BindingType.FREE:
                    scope.add_binding(name, node, BindingType.LOCAL, kind="assignment")
                else:
                    scope.bindings[name].add_definition(node, kind="assignment")
            else:
                scope.add_use(name, node, context="store")
                if name in scope.bindings:
                    scope.bindings[name].add_definition(node, kind="assignment")

        elif isinstance(ctx, ast.Del):
            scope.add_use(name, node, context="del")

        elif isinstance(ctx, ast.Load):
            scope.add_use(name, node, context="load")

    def _visit_lambda(self, node: ast.Lambda, scope: Scope) -> None:
        # Default values in enclosing scope
        self._visit_defaults(node.args, scope)

        lambda_scope = Scope(
            name="<lambda>",
            scope_type=ScopeType.LAMBDA,
            node=node,
        )
        scope.add_child(lambda_scope)
        self.scope_tree.register_node(node, lambda_scope)

        self._bind_parameters(node.args, lambda_scope)
        self._visit_expression(node.body, lambda_scope)

    # ------------------------------------------------------------------
    # Target visitors (assignment targets)
    # ------------------------------------------------------------------

    def _visit_target(
        self, node: ast.AST, scope: Scope, context: str = "store"
    ) -> None:
        if isinstance(node, ast.Name):
            if context == "store":
                if (
                    node.id not in scope.explicit_global_names
                    and node.id not in scope.explicit_nonlocal_names
                ):
                    scope.add_binding(
                        node.id, node, BindingType.LOCAL, kind="assignment"
                    )
                else:
                    if node.id in scope.bindings:
                        scope.bindings[node.id].add_definition(node, kind="assignment")
                    scope.add_use(node.id, node, context="store")
            elif context == "del":
                scope.add_use(node.id, node, context="del")

        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._visit_target(elt, scope, context)

        elif isinstance(node, ast.Starred):
            self._visit_target(node.value, scope, context)

        elif isinstance(node, ast.Attribute):
            self._visit_expression(node.value, scope)

        elif isinstance(node, ast.Subscript):
            self._visit_expression(node.value, scope)
            self._visit_expression(node.slice, scope)

    # ------------------------------------------------------------------
    # Parameter binding
    # ------------------------------------------------------------------

    def _bind_parameters(self, args: ast.arguments, scope: Scope) -> None:
        # positional-only
        for arg in args.posonlyargs:
            self._bind_single_param(
                arg, scope, ParameterKind.POSITIONAL_ONLY
            )

        # positional-or-keyword
        for arg in args.args:
            self._bind_single_param(
                arg, scope, ParameterKind.POSITIONAL_OR_KEYWORD
            )

        # *args
        if args.vararg is not None:
            self._bind_single_param(
                args.vararg, scope, ParameterKind.VAR_POSITIONAL
            )

        # keyword-only
        for arg in args.kwonlyargs:
            self._bind_single_param(
                arg, scope, ParameterKind.KEYWORD_ONLY
            )

        # **kwargs
        if args.kwarg is not None:
            self._bind_single_param(
                args.kwarg, scope, ParameterKind.VAR_KEYWORD
            )

    def _bind_single_param(
        self, arg: ast.arg, scope: Scope, kind: ParameterKind
    ) -> None:
        bi = scope.add_binding(
            arg.arg,
            arg,
            BindingType.PARAMETER,
            kind="parameter",
            is_parameter=True,
            parameter_kind=kind,
        )
        if arg.annotation is not None:
            bi.is_annotated = True
            bi.annotation = arg.annotation
            self._visit_expression(arg.annotation, scope)

    def _visit_defaults(self, args: ast.arguments, scope: Scope) -> None:
        for default in args.defaults:
            self._visit_expression(default, scope)
        for default in args.kw_defaults:
            if default is not None:
                self._visit_expression(default, scope)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _contains_yield(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_source(source: str, filename: str = "<string>") -> ScopeTree:
    """Parse Python source and return a fully-analyzed ScopeTree."""
    tree = ast.parse(source, filename=filename)
    analyzer = ScopeAnalyzer()
    return analyzer.analyze(tree)


def analyze_and_validate(
    source: str, filename: str = "<string>"
) -> Tuple[ScopeTree, List[ScopeDiagnostic]]:
    """Analyze source and run validation, returning tree + diagnostics."""
    scope_tree = analyze_source(source, filename)
    validator = ScopeValidator(scope_tree)
    diagnostics = validator.validate()
    scope_tree.diagnostics.extend(diagnostics)
    return scope_tree, diagnostics


def analyze_closures(source: str, filename: str = "<string>") -> List[Closure]:
    """Analyze source and return detected closures."""
    scope_tree = analyze_source(source, filename)
    closure_analyzer = ClosureAnalyzer(scope_tree)
    return closure_analyzer.detect_closures()


def get_statistics(source: str, filename: str = "<string>") -> ScopeStatistics:
    """Analyze source and return scope statistics."""
    scope_tree = analyze_source(source, filename)
    return ScopeStatistics.from_scope_tree(scope_tree)
