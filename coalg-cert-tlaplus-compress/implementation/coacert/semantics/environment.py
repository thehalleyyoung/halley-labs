"""
Evaluation environment for TLA-lite expressions.

The ``Environment`` provides scoped variable bindings, constant resolution,
operator definitions, and access to the built-in operator registry.

Scopes form a stack: entering a LET/IN block or a quantifier pushes a new
scope; leaving pops it.  Name resolution walks outward from the innermost
scope.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from .values import TLAValue, TLAValueError, SetValue

if TYPE_CHECKING:
    from .state import TLAState


# ---------------------------------------------------------------------------
# AST placeholders – lightweight node types used until a full parser exists
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpDef:
    """Lightweight representation of an operator definition.

    Attributes:
        name:   Operator name (e.g. ``"TypeOK"``).
        params: Formal parameter names (empty for nullary operators).
        body:   The expression AST node (opaque to the environment).
    """
    name: str
    params: Tuple[str, ...] = ()
    body: Any = None
    is_recursive: bool = False


@dataclass(frozen=True)
class ConstantDecl:
    """A CONSTANT declaration, possibly with an assigned value."""
    name: str
    value: Optional[TLAValue] = None
    is_operator: bool = False
    arity: int = 0


# ---------------------------------------------------------------------------
# Scope – one layer of bindings
# ---------------------------------------------------------------------------
class _Scope:
    """A single scope level in the environment stack."""

    __slots__ = ("_bindings", "_label")

    def __init__(self, label: str = "") -> None:
        self._bindings: Dict[str, TLAValue] = {}
        self._label = label

    def bind(self, name: str, value: TLAValue) -> None:
        self._bindings[name] = value

    def lookup(self, name: str) -> Optional[TLAValue]:
        return self._bindings.get(name)

    def has(self, name: str) -> bool:
        return name in self._bindings

    @property
    def names(self) -> FrozenSet[str]:
        return frozenset(self._bindings.keys())

    @property
    def label(self) -> str:
        return self._label

    def __repr__(self) -> str:
        return f"_Scope({self._label}, {set(self._bindings.keys())})"


# ---------------------------------------------------------------------------
# BuiltinEntry – one built-in operator
# ---------------------------------------------------------------------------
@dataclass
class BuiltinEntry:
    """Registry entry for a built-in operator or module operator."""
    name: str
    module: str
    arity: int
    evaluator: Callable[..., TLAValue]
    is_lazy: bool = False  # True if arguments should not be pre-evaluated


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Environment:
    """Scoped binding environment for TLA+ expression evaluation.

    The environment maintains:
    * A stack of scopes (innermost-first) for local bindings.
    * A constant table mapping CONSTANT names to assigned values.
    * An operator table mapping operator names to ``OpDef`` nodes.
    * A built-in registry mapping (module, name) to ``BuiltinEntry``.
    """

    def __init__(self) -> None:
        self._scopes: List[_Scope] = [_Scope("global")]
        self._constants: Dict[str, ConstantDecl] = {}
        self._operators: Dict[str, OpDef] = {}
        self._builtins: Dict[str, BuiltinEntry] = {}
        self._module_builtins: Dict[Tuple[str, str], BuiltinEntry] = {}
        self._imported_modules: Set[str] = set()

    # --- scope management -------------------------------------------------

    def push_scope(self, label: str = "") -> None:
        self._scopes.append(_Scope(label))

    def pop_scope(self) -> None:
        if len(self._scopes) <= 1:
            raise TLAValueError("Cannot pop the global scope")
        self._scopes.pop()

    @property
    def scope_depth(self) -> int:
        return len(self._scopes)

    @property
    def current_scope_label(self) -> str:
        return self._scopes[-1].label

    # --- variable binding -------------------------------------------------

    def bind(self, name: str, value: TLAValue) -> None:
        """Bind *name* in the current (innermost) scope."""
        self._scopes[-1].bind(name, value)

    def bind_in_global(self, name: str, value: TLAValue) -> None:
        """Bind *name* directly in the global scope."""
        self._scopes[0].bind(name, value)

    def lookup(self, name: str) -> Optional[TLAValue]:
        """Resolve *name* by walking outward through scopes."""
        for scope in reversed(self._scopes):
            val = scope.lookup(name)
            if val is not None:
                return val
        return None

    def resolve(self, name: str) -> TLAValue:
        """Like ``lookup`` but raises if not found."""
        val = self.lookup(name)
        if val is None:
            # Check constants
            cdecl = self._constants.get(name)
            if cdecl is not None and cdecl.value is not None:
                return cdecl.value
            raise TLAValueError(
                f"Unresolved name '{name}'. "
                f"Scopes: {[s.label for s in self._scopes]}"
            )
        return val

    def has_binding(self, name: str) -> bool:
        return self.lookup(name) is not None or (
            name in self._constants and self._constants[name].value is not None
        )

    # --- constants --------------------------------------------------------

    def declare_constant(self, name: str, value: Optional[TLAValue] = None,
                         is_operator: bool = False, arity: int = 0) -> None:
        self._constants[name] = ConstantDecl(
            name=name, value=value, is_operator=is_operator, arity=arity
        )

    def assign_constant(self, name: str, value: TLAValue) -> None:
        if name not in self._constants:
            self._constants[name] = ConstantDecl(name=name, value=value)
        else:
            old = self._constants[name]
            self._constants[name] = ConstantDecl(
                name=name, value=value,
                is_operator=old.is_operator, arity=old.arity
            )

    def get_constant(self, name: str) -> Optional[ConstantDecl]:
        return self._constants.get(name)

    def constant_value(self, name: str) -> Optional[TLAValue]:
        cdecl = self._constants.get(name)
        return cdecl.value if cdecl else None

    def unresolved_constants(self) -> List[str]:
        return [c.name for c in self._constants.values() if c.value is None]

    # --- operator definitions ---------------------------------------------

    def define_operator(self, op: OpDef) -> None:
        self._operators[op.name] = op

    def get_operator(self, name: str) -> Optional[OpDef]:
        return self._operators.get(name)

    def has_operator(self, name: str) -> bool:
        return name in self._operators

    def operator_names(self) -> FrozenSet[str]:
        return frozenset(self._operators.keys())

    # --- built-in registry ------------------------------------------------

    def register_builtin(self, entry: BuiltinEntry) -> None:
        self._builtins[entry.name] = entry
        self._module_builtins[(entry.module, entry.name)] = entry

    def get_builtin(self, name: str) -> Optional[BuiltinEntry]:
        return self._builtins.get(name)

    def get_module_builtin(self, module: str, name: str) -> Optional[BuiltinEntry]:
        return self._module_builtins.get((module, name))

    def import_module(self, module_name: str) -> None:
        """Record that *module_name* has been EXTENDed / INSTANCEd."""
        self._imported_modules.add(module_name)

    def is_module_imported(self, module_name: str) -> bool:
        return module_name in self._imported_modules

    def all_builtins(self) -> Dict[str, BuiltinEntry]:
        return dict(self._builtins)

    # --- context manager for scoped bindings ------------------------------

    class _ScopeCtx:
        def __init__(self, env: "Environment", label: str) -> None:
            self._env = env
            self._label = label

        def __enter__(self) -> "Environment":
            self._env.push_scope(self._label)
            return self._env

        def __exit__(self, *exc: Any) -> None:
            self._env.pop_scope()

    def scope(self, label: str = "") -> _ScopeCtx:
        """Usage:  ``with env.scope("let"): env.bind(...)``"""
        return self._ScopeCtx(self, label)

    # --- snapshot / clone -------------------------------------------------

    def snapshot(self) -> "Environment":
        """Cheap shallow copy (scopes are shared until mutation)."""
        env = Environment()
        env._scopes = [copy(s) for s in self._scopes]
        env._constants = dict(self._constants)
        env._operators = dict(self._operators)
        env._builtins = self._builtins
        env._module_builtins = self._module_builtins
        env._imported_modules = set(self._imported_modules)
        return env

    # --- diagnostics ------------------------------------------------------

    def dump(self) -> str:
        lines: List[str] = ["Environment:"]
        for i, scope in enumerate(reversed(self._scopes)):
            lines.append(f"  scope[{i}] ({scope.label}): {sorted(scope.names)}")
        if self._constants:
            lines.append(f"  constants: {sorted(self._constants.keys())}")
        if self._operators:
            lines.append(f"  operators: {sorted(self._operators.keys())}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Environment(scopes={len(self._scopes)}, "
            f"consts={len(self._constants)}, ops={len(self._operators)})"
        )
