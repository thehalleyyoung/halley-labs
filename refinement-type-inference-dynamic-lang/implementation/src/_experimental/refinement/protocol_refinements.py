"""Refinement types for Python's structural subtyping (Protocols).

Defines protocol definitions, compliance checking, function refinements
with pre/post conditions, and inference of protocols from usage patterns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from src.heap.heap_model import HeapAddress, AbstractValue, AbstractHeap
from src.refinement.python_refinements import (
    HeapPredicate, HeapPredKind, PyRefinementType, PyType,
    ProtocolType, FunctionPyType, ClassType, AnyType, NeverType,
    PyUnionType, IntPyType, StrPyType, BoolPyType, NoneType as NoneRefType,
    ListPyType, DictPyType,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BuiltinProtocol(Enum):
    """Well-known protocols from ``typing`` / ``collections.abc``."""
    SIZED = auto()
    ITERABLE = auto()
    ITERATOR = auto()
    CONTAINER = auto()
    HASHABLE = auto()
    REVERSIBLE = auto()
    SEQUENCE = auto()
    MUTABLE_SEQUENCE = auto()
    MAPPING = auto()
    MUTABLE_MAPPING = auto()
    SET_ = auto()
    MUTABLE_SET = auto()
    CALLABLE_ = auto()
    AWAITABLE = auto()
    ASYNC_ITERABLE = auto()
    ASYNC_ITERATOR = auto()
    CONTEXT_MANAGER = auto()
    ASYNC_CONTEXT_MANAGER = auto()
    SUPPORTS_INT = auto()
    SUPPORTS_FLOAT = auto()
    SUPPORTS_COMPLEX = auto()
    SUPPORTS_BYTES = auto()
    SUPPORTS_ABS = auto()
    SUPPORTS_ROUND = auto()
    BUFFER = auto()


# ---------------------------------------------------------------------------
# FunctionSignature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionSignature:
    """Expected method signature inside a protocol."""
    params: Tuple[Tuple[str, PyType], ...] = ()
    return_type: PyType = field(default_factory=AnyType)
    is_property: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False

    # Convenience constructor accepting a list
    @classmethod
    def of(
        cls,
        params: List[Tuple[str, PyType]],
        return_type: PyType,
        *,
        is_property: bool = False,
        is_classmethod: bool = False,
        is_staticmethod: bool = False,
    ) -> FunctionSignature:
        return cls(
            params=tuple(params),
            return_type=return_type,
            is_property=is_property,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
        )

    def compatible_with(self, other: FunctionSignature) -> bool:
        """Check structural compatibility (covariant return, contravariant params)."""
        if self.is_property != other.is_property:
            return False
        if self.is_classmethod != other.is_classmethod:
            return False
        if self.is_staticmethod != other.is_staticmethod:
            return False
        # Parameter count must match (ignoring `self`)
        self_params = self.params
        other_params = other.params
        if len(self_params) != len(other_params):
            return False
        # Contravariant check on parameters: other params should be subtypes
        # of self params (the implementation accepts at least what the protocol
        # demands).  We use a simplified structural check here.
        for (_, p_ty), (_, o_ty) in zip(self_params, other_params):
            if not _is_subtype(p_ty, o_ty):
                return False
        # Covariant check on return type
        if not _is_subtype(self.return_type, other.return_type):
            if not _is_subtype(other.return_type, self.return_type):
                return False
        return True

    def pretty(self) -> str:
        prefix = ""
        if self.is_property:
            prefix = "@property "
        elif self.is_classmethod:
            prefix = "@classmethod "
        elif self.is_staticmethod:
            prefix = "@staticmethod "
        params_str = ", ".join(
            f"{name}: {_type_name(ty)}" for name, ty in self.params
        )
        return f"{prefix}({params_str}) -> {_type_name(self.return_type)}"


# ---------------------------------------------------------------------------
# ProtocolComplianceResult
# ---------------------------------------------------------------------------

@dataclass
class ProtocolComplianceResult:
    """Result of checking whether a class satisfies a protocol."""
    compliant: bool
    protocol_name: str
    missing_methods: Set[str] = field(default_factory=set)
    missing_attrs: Set[str] = field(default_factory=set)
    incompatible_methods: Dict[str, str] = field(default_factory=dict)
    incompatible_attrs: Dict[str, str] = field(default_factory=dict)
    extra_methods: Set[str] = field(default_factory=set)

    def error_messages(self) -> List[str]:
        msgs: List[str] = []
        for m in sorted(self.missing_methods):
            msgs.append(f"Missing method '{m}' required by {self.protocol_name}")
        for a in sorted(self.missing_attrs):
            msgs.append(f"Missing attribute '{a}' required by {self.protocol_name}")
        for m, err in sorted(self.incompatible_methods.items()):
            msgs.append(
                f"Method '{m}' incompatible with {self.protocol_name}: {err}"
            )
        for a, err in sorted(self.incompatible_attrs.items()):
            msgs.append(
                f"Attribute '{a}' incompatible with {self.protocol_name}: {err}"
            )
        return msgs

    def pretty(self) -> str:
        if self.compliant:
            return f"✓ Compliant with {self.protocol_name}"
        lines = [f"✗ Not compliant with {self.protocol_name}:"]
        lines.extend(f"  - {m}" for m in self.error_messages())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProtocolDefinition
# ---------------------------------------------------------------------------

@dataclass
class ProtocolDefinition:
    """Describes what a protocol requires of conforming classes."""
    name: str
    required_methods: Dict[str, FunctionSignature] = field(default_factory=dict)
    required_attrs: Dict[str, PyType] = field(default_factory=dict)
    required_properties: Dict[str, PyType] = field(default_factory=dict)
    is_runtime_checkable: bool = False
    covariant_params: Set[str] = field(default_factory=set)
    contravariant_params: Set[str] = field(default_factory=set)

    def is_satisfied_by(
        self,
        cls_type: ClassType,
        registry: Optional[BuiltinProtocolRegistry] = None,
    ) -> ProtocolComplianceResult:
        missing_m = self.get_missing(cls_type)
        incompat_m = self.get_incompatible(cls_type)
        missing_a: Set[str] = set()
        incompat_a: Dict[str, str] = {}

        cls_attrs = getattr(cls_type, "attrs", {}) or {}
        for attr, expected_ty in self.required_attrs.items():
            if attr not in cls_attrs:
                missing_a.add(attr)
            else:
                actual = cls_attrs[attr]
                if not _is_subtype(actual, expected_ty):
                    incompat_a[attr] = (
                        f"expected {_type_name(expected_ty)}, "
                        f"got {_type_name(actual)}"
                    )

        for prop, expected_ty in self.required_properties.items():
            if prop not in cls_attrs:
                missing_a.add(prop)

        cls_methods = getattr(cls_type, "methods", {}) or {}
        extra = set(cls_methods.keys()) - set(self.required_methods.keys())

        compliant = not missing_m and not missing_a and not incompat_m and not incompat_a
        return ProtocolComplianceResult(
            compliant=compliant,
            protocol_name=self.name,
            missing_methods=missing_m,
            missing_attrs=missing_a,
            incompatible_methods=incompat_m,
            incompatible_attrs=incompat_a,
            extra_methods=extra,
        )

    def get_missing(self, cls_type: ClassType) -> Set[str]:
        cls_methods = getattr(cls_type, "methods", {}) or {}
        missing: Set[str] = set()
        for method_name in self.required_methods:
            if method_name not in cls_methods:
                missing.add(method_name)
        return missing

    def get_incompatible(self, cls_type: ClassType) -> Dict[str, str]:
        cls_methods = getattr(cls_type, "methods", {}) or {}
        result: Dict[str, str] = {}
        for method_name, expected_sig in self.required_methods.items():
            if method_name in cls_methods:
                actual = cls_methods[method_name]
                actual_sig = _to_function_signature(actual)
                if actual_sig is not None and not expected_sig.compatible_with(actual_sig):
                    result[method_name] = (
                        f"expected {expected_sig.pretty()}, "
                        f"got {actual_sig.pretty()}"
                    )
        return result

    def join(self, other: ProtocolDefinition) -> ProtocolDefinition:
        """Lattice join: intersection of requirements (less demanding)."""
        common_methods: Dict[str, FunctionSignature] = {}
        for name in self.required_methods:
            if name in other.required_methods:
                common_methods[name] = self.required_methods[name]
        common_attrs: Dict[str, PyType] = {}
        for name in self.required_attrs:
            if name in other.required_attrs:
                common_attrs[name] = self.required_attrs[name]
        common_props: Dict[str, PyType] = {}
        for name in self.required_properties:
            if name in other.required_properties:
                common_props[name] = self.required_properties[name]
        return ProtocolDefinition(
            name=f"({self.name} | {other.name})",
            required_methods=common_methods,
            required_attrs=common_attrs,
            required_properties=common_props,
            is_runtime_checkable=self.is_runtime_checkable and other.is_runtime_checkable,
            covariant_params=self.covariant_params & other.covariant_params,
            contravariant_params=self.contravariant_params & other.contravariant_params,
        )

    def meet(self, other: ProtocolDefinition) -> ProtocolDefinition:
        """Lattice meet: union of requirements (more demanding)."""
        merged_methods = {**self.required_methods, **other.required_methods}
        merged_attrs = {**self.required_attrs, **other.required_attrs}
        merged_props = {**self.required_properties, **other.required_properties}
        return ProtocolDefinition(
            name=f"({self.name} & {other.name})",
            required_methods=merged_methods,
            required_attrs=merged_attrs,
            required_properties=merged_props,
            is_runtime_checkable=self.is_runtime_checkable or other.is_runtime_checkable,
            covariant_params=self.covariant_params | other.covariant_params,
            contravariant_params=self.contravariant_params | other.contravariant_params,
        )

    def pretty(self) -> str:
        lines = [f"protocol {self.name}:"]
        for m, sig in sorted(self.required_methods.items()):
            lines.append(f"  def {m}{sig.pretty()}")
        for a, ty in sorted(self.required_attrs.items()):
            lines.append(f"  {a}: {_type_name(ty)}")
        for p, ty in sorted(self.required_properties.items()):
            lines.append(f"  @property {p}: {_type_name(ty)}")
        if self.is_runtime_checkable:
            lines.append("  [runtime_checkable]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FunctionRefinement
# ---------------------------------------------------------------------------

@dataclass
class FunctionRefinement:
    """Dependent function type with pre/post conditions and frame."""
    params: List[Tuple[str, PyRefinementType]] = field(default_factory=list)
    varargs: Optional[PyRefinementType] = None
    kwargs: Optional[PyRefinementType] = None
    return_type: PyRefinementType = field(default_factory=lambda: PyRefinementType(base=AnyType()))
    raises: FrozenSet[str] = field(default_factory=frozenset)
    pre_conditions: List[HeapPredicate] = field(default_factory=list)
    post_conditions: List[HeapPredicate] = field(default_factory=list)
    frame: FrozenSet[str] = field(default_factory=frozenset)
    is_pure: bool = False
    is_generator: bool = False
    is_async: bool = False

    def check_preconditions(
        self, args: Dict[str, Any], heap: AbstractHeap
    ) -> List[str]:
        """Return list of violated precondition descriptions."""
        violations: List[str] = []
        for pre in self.pre_conditions:
            kind = getattr(pre, "kind", None)
            if kind == HeapPredKind.NOT_NULL:
                var = getattr(pre, "var", None)
                if var and var in args and args[var] is None:
                    violations.append(f"Precondition violated: {var} must not be None")
            elif kind == HeapPredKind.TYPE_IS:
                var = getattr(pre, "var", None)
                expected = getattr(pre, "type_name", None)
                if var and var in args and expected:
                    actual_type = type(args[var]).__name__
                    if actual_type != expected:
                        violations.append(
                            f"Precondition violated: {var} must be {expected}, "
                            f"got {actual_type}"
                        )
            else:
                # Generic predicate – we can only flag if we know it fails
                pass
        return violations

    def apply_postconditions(
        self, result_var: str, env: Dict[str, PyRefinementType]
    ) -> Dict[str, PyRefinementType]:
        """Apply postconditions to produce an updated environment."""
        new_env = dict(env)
        new_env[result_var] = self.return_type
        for post in self.post_conditions:
            var = getattr(post, "var", None)
            ref = getattr(post, "refinement", None)
            if var and ref:
                new_env[var] = ref
        return new_env

    def compose(self, other: FunctionRefinement) -> FunctionRefinement:
        """Sequential composition: apply *self* then *other*."""
        return FunctionRefinement(
            params=self.params,
            varargs=self.varargs,
            kwargs=self.kwargs,
            return_type=other.return_type,
            raises=self.raises | other.raises,
            pre_conditions=list(self.pre_conditions),
            post_conditions=list(other.post_conditions),
            frame=self.frame | other.frame,
            is_pure=self.is_pure and other.is_pure,
            is_generator=self.is_generator or other.is_generator,
            is_async=self.is_async or other.is_async,
        )

    def join(self, other: FunctionRefinement) -> FunctionRefinement:
        """Lattice join (upper bound): weaker specification."""
        # Merge param lists by position, widening types
        merged_params: List[Tuple[str, PyRefinementType]] = []
        for i in range(min(len(self.params), len(other.params))):
            name = self.params[i][0]
            merged_params.append((name, _join_ref(self.params[i][1], other.params[i][1])))
        return FunctionRefinement(
            params=merged_params,
            varargs=self.varargs if self.varargs else other.varargs,
            kwargs=self.kwargs if self.kwargs else other.kwargs,
            return_type=_join_ref(self.return_type, other.return_type),
            raises=self.raises | other.raises,
            pre_conditions=[p for p in self.pre_conditions if p in other.pre_conditions],
            post_conditions=[p for p in self.post_conditions if p in other.post_conditions],
            frame=self.frame | other.frame,
            is_pure=self.is_pure and other.is_pure,
            is_generator=self.is_generator and other.is_generator,
            is_async=self.is_async and other.is_async,
        )

    def pretty(self) -> str:
        parts: List[str] = []
        if self.is_async:
            parts.append("async ")
        if self.is_generator:
            parts.append("gen ")
        params_str = ", ".join(
            f"{n}: {getattr(t, 'pretty', lambda: str(t))()}" for n, t in self.params
        )
        if self.varargs:
            params_str += f", *args: {getattr(self.varargs, 'pretty', lambda: '...')()}"
        if self.kwargs:
            params_str += f", **kw: {getattr(self.kwargs, 'pretty', lambda: '...')()}"
        ret = getattr(self.return_type, "pretty", lambda: str(self.return_type))()
        parts.append(f"({params_str}) -> {ret}")
        if self.raises:
            parts.append(f" raises {{{', '.join(sorted(self.raises))}}}")
        if self.is_pure:
            parts.append(" [pure]")
        if self.pre_conditions:
            parts.append(f" pre({len(self.pre_conditions)})")
        if self.post_conditions:
            parts.append(f" post({len(self.post_conditions)})")
        return "".join(parts)


# ---------------------------------------------------------------------------
# ProtocolRefinement
# ---------------------------------------------------------------------------

@dataclass
class ProtocolRefinement:
    """Refinement type layered on top of a protocol definition."""
    protocol: ProtocolDefinition
    method_refinements: Dict[str, FunctionRefinement] = field(default_factory=dict)
    attr_refinements: Dict[str, PyRefinementType] = field(default_factory=dict)

    def with_method_refinement(
        self, method_name: str, refinement: FunctionRefinement
    ) -> ProtocolRefinement:
        new_methods = {**self.method_refinements, method_name: refinement}
        return ProtocolRefinement(
            protocol=self.protocol,
            method_refinements=new_methods,
            attr_refinements=dict(self.attr_refinements),
        )

    def with_attr_refinement(
        self, attr_name: str, refinement: PyRefinementType
    ) -> ProtocolRefinement:
        new_attrs = {**self.attr_refinements, attr_name: refinement}
        return ProtocolRefinement(
            protocol=self.protocol,
            method_refinements=dict(self.method_refinements),
            attr_refinements=new_attrs,
        )

    def join(self, other: ProtocolRefinement) -> ProtocolRefinement:
        merged_proto = self.protocol.join(other.protocol)
        merged_meths: Dict[str, FunctionRefinement] = {}
        for name in self.method_refinements:
            if name in other.method_refinements:
                merged_meths[name] = self.method_refinements[name].join(
                    other.method_refinements[name]
                )
        merged_attrs: Dict[str, PyRefinementType] = {}
        for name in self.attr_refinements:
            if name in other.attr_refinements:
                merged_attrs[name] = _join_ref(
                    self.attr_refinements[name], other.attr_refinements[name]
                )
        return ProtocolRefinement(
            protocol=merged_proto,
            method_refinements=merged_meths,
            attr_refinements=merged_attrs,
        )

    def meet(self, other: ProtocolRefinement) -> ProtocolRefinement:
        merged_proto = self.protocol.meet(other.protocol)
        merged_meths = {**self.method_refinements, **other.method_refinements}
        merged_attrs = {**self.attr_refinements, **other.attr_refinements}
        return ProtocolRefinement(
            protocol=merged_proto,
            method_refinements=merged_meths,
            attr_refinements=merged_attrs,
        )

    def to_predicates(self, var: str) -> List[HeapPredicate]:
        preds: List[HeapPredicate] = []
        for method_name in self.protocol.required_methods:
            preds.append(HeapPredicate(
                kind=HeapPredKind.HAS_ATTR,
                var=var,
                attr=method_name,
            ))
        for attr_name in self.protocol.required_attrs:
            preds.append(HeapPredicate(
                kind=HeapPredKind.HAS_ATTR,
                var=var,
                attr=attr_name,
            ))
        return preds

    def pretty(self) -> str:
        lines = [f"ProtocolRefinement({self.protocol.name})"]
        for m, ref in sorted(self.method_refinements.items()):
            lines.append(f"  {m}: {ref.pretty()}")
        for a, ref in sorted(self.attr_refinements.items()):
            lines.append(f"  {a}: {getattr(ref, 'pretty', lambda: str(ref))()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ContextManagerRefinement
# ---------------------------------------------------------------------------

@dataclass
class ContextManagerRefinement:
    """Special refinement tracking context-manager semantics."""
    enter_type: PyRefinementType = field(
        default_factory=lambda: PyRefinementType(base=AnyType())
    )
    exit_suppresses: Optional[Set[str]] = None
    resource_type: Optional[PyType] = None
    is_reentrant: bool = False
    is_reusable: bool = False

    def after_enter(self, var_name: str) -> PyRefinementType:
        """Return the refinement type that *var_name* receives after ``__enter__``."""
        return self.enter_type

    def after_exit(self, exc_type: Optional[str]) -> bool:
        """Return ``True`` if the given exception type would be suppressed."""
        if self.exit_suppresses is None:
            return False
        if exc_type is None:
            return True
        return exc_type in self.exit_suppresses

    def check_resource_leak(self, used_in_with: bool) -> Optional[str]:
        """Warn when a resource is acquired without a ``with`` block."""
        if self.resource_type is not None and not used_in_with:
            return (
                f"Resource of type {_type_name(self.resource_type)} acquired "
                f"outside a 'with' statement – possible resource leak"
            )
        return None

    def pretty(self) -> str:
        parts = [f"ContextManager[{getattr(self.enter_type, 'pretty', lambda: '?')()}]"]
        if self.exit_suppresses:
            parts.append(f"suppresses={{{', '.join(sorted(self.exit_suppresses))}}}")
        if self.resource_type:
            parts.append(f"resource={_type_name(self.resource_type)}")
        flags: List[str] = []
        if self.is_reentrant:
            flags.append("reentrant")
        if self.is_reusable:
            flags.append("reusable")
        if flags:
            parts.append("[" + ", ".join(flags) + "]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# DecoratorEffect
# ---------------------------------------------------------------------------

@dataclass
class DecoratorEffect:
    """Describes how a decorator transforms a function refinement."""
    decorator_name: str

    def transform_signature(self, original: FunctionRefinement) -> FunctionRefinement:
        handler = _DECORATOR_HANDLERS.get(self.decorator_name)
        if handler is not None:
            return handler(original)
        # Unknown decorator – pass through unchanged
        return original


def _transform_property(fr: FunctionRefinement) -> FunctionRefinement:
    """@property: method(self) -> T  →  property descriptor returning T."""
    return FunctionRefinement(
        params=[],
        return_type=fr.return_type,
        is_pure=True,
        pre_conditions=fr.pre_conditions,
        post_conditions=fr.post_conditions,
        frame=frozenset(),
    )


def _transform_classmethod(fr: FunctionRefinement) -> FunctionRefinement:
    """@classmethod: insert ``cls`` as the first parameter."""
    cls_param = ("cls", PyRefinementType(base=AnyType()))
    new_params = [cls_param] + fr.params
    return FunctionRefinement(
        params=new_params,
        varargs=fr.varargs,
        kwargs=fr.kwargs,
        return_type=fr.return_type,
        raises=fr.raises,
        pre_conditions=fr.pre_conditions,
        post_conditions=fr.post_conditions,
        frame=fr.frame,
        is_pure=fr.is_pure,
        is_generator=fr.is_generator,
        is_async=fr.is_async,
    )


def _transform_staticmethod(fr: FunctionRefinement) -> FunctionRefinement:
    """@staticmethod: remove ``self`` from parameters."""
    new_params = [p for p in fr.params if p[0] != "self"]
    return FunctionRefinement(
        params=new_params,
        varargs=fr.varargs,
        kwargs=fr.kwargs,
        return_type=fr.return_type,
        raises=fr.raises,
        pre_conditions=fr.pre_conditions,
        post_conditions=fr.post_conditions,
        frame=fr.frame,
        is_pure=fr.is_pure,
        is_generator=fr.is_generator,
        is_async=fr.is_async,
    )


def _transform_abstractmethod(fr: FunctionRefinement) -> FunctionRefinement:
    """@abstractmethod: pass-through, the method body is ignored."""
    return fr


def _transform_wraps(fr: FunctionRefinement) -> FunctionRefinement:
    """@functools.wraps: preserves the original signature."""
    return fr


def _transform_lru_cache(fr: FunctionRefinement) -> FunctionRefinement:
    """@functools.lru_cache: add cache semantics, mark pure after first call."""
    return FunctionRefinement(
        params=fr.params,
        varargs=fr.varargs,
        kwargs=fr.kwargs,
        return_type=fr.return_type,
        raises=fr.raises,
        pre_conditions=fr.pre_conditions,
        post_conditions=fr.post_conditions,
        frame=frozenset(),
        is_pure=True,
        is_generator=fr.is_generator,
        is_async=fr.is_async,
    )


def _transform_contextmanager(fr: FunctionRefinement) -> FunctionRefinement:
    """@contextlib.contextmanager: generator -> context manager."""
    return FunctionRefinement(
        params=fr.params,
        varargs=fr.varargs,
        kwargs=fr.kwargs,
        return_type=PyRefinementType(base=AnyType()),
        raises=fr.raises,
        pre_conditions=fr.pre_conditions,
        post_conditions=fr.post_conditions,
        frame=fr.frame,
        is_pure=False,
        is_generator=False,
        is_async=fr.is_async,
    )


_DECORATOR_HANDLERS: Dict[str, Any] = {
    "property": _transform_property,
    "classmethod": _transform_classmethod,
    "staticmethod": _transform_staticmethod,
    "abstractmethod": _transform_abstractmethod,
    "functools.wraps": _transform_wraps,
    "functools.lru_cache": _transform_lru_cache,
    "contextmanager": _transform_contextmanager,
    "contextlib.contextmanager": _transform_contextmanager,
}


# ---------------------------------------------------------------------------
# BuiltinProtocolRegistry
# ---------------------------------------------------------------------------

class BuiltinProtocolRegistry:
    """Registry of Python's built-in protocols from ``collections.abc``."""

    def __init__(self) -> None:
        self.protocols: Dict[BuiltinProtocol, ProtocolDefinition] = {}
        self.init_builtin_protocols()

    # ---- public API -------------------------------------------------------

    def get_protocol(self, name: BuiltinProtocol) -> ProtocolDefinition:
        return self.protocols[name]

    def check_all_protocols(
        self, cls_type: ClassType
    ) -> Dict[BuiltinProtocol, ProtocolComplianceResult]:
        results: Dict[BuiltinProtocol, ProtocolComplianceResult] = {}
        for bp, pdef in self.protocols.items():
            results[bp] = pdef.is_satisfied_by(cls_type, self)
        return results

    def infer_protocols(self, cls_type: ClassType) -> Set[BuiltinProtocol]:
        satisfied: Set[BuiltinProtocol] = set()
        for bp, pdef in self.protocols.items():
            result = pdef.is_satisfied_by(cls_type, self)
            if result.compliant:
                satisfied.add(bp)
        return satisfied

    def get_refined_protocol(
        self,
        proto: BuiltinProtocol,
        refinements: Optional[Dict[str, FunctionRefinement]] = None,
    ) -> ProtocolRefinement:
        pdef = self.protocols[proto]
        method_refs = refinements if refinements else {}
        return ProtocolRefinement(protocol=pdef, method_refinements=method_refs)

    # ---- initialisation ---------------------------------------------------

    def init_builtin_protocols(self) -> None:
        any_ty = AnyType()
        bool_ty = BoolPyType()
        int_ty = IntPyType()
        none_ty = NoneRefType()

        self_param = ("self", any_ty)

        # -- Sized --
        self.protocols[BuiltinProtocol.SIZED] = ProtocolDefinition(
            name="Sized",
            required_methods={
                "__len__": FunctionSignature.of([self_param], int_ty),
            },
            is_runtime_checkable=True,
        )

        # -- Hashable --
        self.protocols[BuiltinProtocol.HASHABLE] = ProtocolDefinition(
            name="Hashable",
            required_methods={
                "__hash__": FunctionSignature.of([self_param], int_ty),
            },
            is_runtime_checkable=True,
        )

        # -- Container --
        self.protocols[BuiltinProtocol.CONTAINER] = ProtocolDefinition(
            name="Container",
            required_methods={
                "__contains__": FunctionSignature.of(
                    [self_param, ("item", any_ty)], bool_ty
                ),
            },
            is_runtime_checkable=True,
        )

        # -- Iterable --
        self.protocols[BuiltinProtocol.ITERABLE] = ProtocolDefinition(
            name="Iterable",
            required_methods={
                "__iter__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- Iterator (extends Iterable) --
        self.protocols[BuiltinProtocol.ITERATOR] = ProtocolDefinition(
            name="Iterator",
            required_methods={
                "__iter__": FunctionSignature.of([self_param], any_ty),
                "__next__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- Reversible --
        self.protocols[BuiltinProtocol.REVERSIBLE] = ProtocolDefinition(
            name="Reversible",
            required_methods={
                "__reversed__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- Callable --
        self.protocols[BuiltinProtocol.CALLABLE_] = ProtocolDefinition(
            name="Callable",
            required_methods={
                "__call__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- Sequence --
        self.protocols[BuiltinProtocol.SEQUENCE] = ProtocolDefinition(
            name="Sequence",
            required_methods={
                "__getitem__": FunctionSignature.of(
                    [self_param, ("index", int_ty)], any_ty
                ),
                "__len__": FunctionSignature.of([self_param], int_ty),
                "__contains__": FunctionSignature.of(
                    [self_param, ("item", any_ty)], bool_ty
                ),
                "__iter__": FunctionSignature.of([self_param], any_ty),
                "__reversed__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=False,
        )

        # -- MutableSequence --
        seq_methods = dict(self.protocols[BuiltinProtocol.SEQUENCE].required_methods)
        seq_methods.update({
            "__setitem__": FunctionSignature.of(
                [self_param, ("index", int_ty), ("value", any_ty)], none_ty
            ),
            "__delitem__": FunctionSignature.of(
                [self_param, ("index", int_ty)], none_ty
            ),
            "insert": FunctionSignature.of(
                [self_param, ("index", int_ty), ("value", any_ty)], none_ty
            ),
        })
        self.protocols[BuiltinProtocol.MUTABLE_SEQUENCE] = ProtocolDefinition(
            name="MutableSequence",
            required_methods=seq_methods,
            is_runtime_checkable=False,
        )

        # -- Mapping --
        self.protocols[BuiltinProtocol.MAPPING] = ProtocolDefinition(
            name="Mapping",
            required_methods={
                "__getitem__": FunctionSignature.of(
                    [self_param, ("key", any_ty)], any_ty
                ),
                "__len__": FunctionSignature.of([self_param], int_ty),
                "__iter__": FunctionSignature.of([self_param], any_ty),
                "__contains__": FunctionSignature.of(
                    [self_param, ("key", any_ty)], bool_ty
                ),
                "keys": FunctionSignature.of([self_param], any_ty),
                "values": FunctionSignature.of([self_param], any_ty),
                "items": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=False,
        )

        # -- MutableMapping --
        map_methods = dict(self.protocols[BuiltinProtocol.MAPPING].required_methods)
        map_methods.update({
            "__setitem__": FunctionSignature.of(
                [self_param, ("key", any_ty), ("value", any_ty)], none_ty
            ),
            "__delitem__": FunctionSignature.of(
                [self_param, ("key", any_ty)], none_ty
            ),
        })
        self.protocols[BuiltinProtocol.MUTABLE_MAPPING] = ProtocolDefinition(
            name="MutableMapping",
            required_methods=map_methods,
            is_runtime_checkable=False,
        )

        # -- Set --
        self.protocols[BuiltinProtocol.SET_] = ProtocolDefinition(
            name="Set",
            required_methods={
                "__contains__": FunctionSignature.of(
                    [self_param, ("item", any_ty)], bool_ty
                ),
                "__iter__": FunctionSignature.of([self_param], any_ty),
                "__len__": FunctionSignature.of([self_param], int_ty),
            },
            is_runtime_checkable=False,
        )

        # -- MutableSet --
        set_methods = dict(self.protocols[BuiltinProtocol.SET_].required_methods)
        set_methods.update({
            "add": FunctionSignature.of(
                [self_param, ("item", any_ty)], none_ty
            ),
            "discard": FunctionSignature.of(
                [self_param, ("item", any_ty)], none_ty
            ),
        })
        self.protocols[BuiltinProtocol.MUTABLE_SET] = ProtocolDefinition(
            name="MutableSet",
            required_methods=set_methods,
            is_runtime_checkable=False,
        )

        # -- Awaitable --
        self.protocols[BuiltinProtocol.AWAITABLE] = ProtocolDefinition(
            name="Awaitable",
            required_methods={
                "__await__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- AsyncIterable --
        self.protocols[BuiltinProtocol.ASYNC_ITERABLE] = ProtocolDefinition(
            name="AsyncIterable",
            required_methods={
                "__aiter__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- AsyncIterator --
        self.protocols[BuiltinProtocol.ASYNC_ITERATOR] = ProtocolDefinition(
            name="AsyncIterator",
            required_methods={
                "__aiter__": FunctionSignature.of([self_param], any_ty),
                "__anext__": FunctionSignature.of([self_param], any_ty),
            },
            is_runtime_checkable=True,
        )

        # -- ContextManager --
        self.protocols[BuiltinProtocol.CONTEXT_MANAGER] = ProtocolDefinition(
            name="ContextManager",
            required_methods={
                "__enter__": FunctionSignature.of([self_param], any_ty),
                "__exit__": FunctionSignature.of(
                    [self_param, ("exc_type", any_ty),
                     ("exc_val", any_ty), ("exc_tb", any_ty)],
                    bool_ty,
                ),
            },
            is_runtime_checkable=True,
        )

        # -- AsyncContextManager --
        self.protocols[BuiltinProtocol.ASYNC_CONTEXT_MANAGER] = ProtocolDefinition(
            name="AsyncContextManager",
            required_methods={
                "__aenter__": FunctionSignature.of([self_param], any_ty),
                "__aexit__": FunctionSignature.of(
                    [self_param, ("exc_type", any_ty),
                     ("exc_val", any_ty), ("exc_tb", any_ty)],
                    bool_ty,
                ),
            },
            is_runtime_checkable=True,
        )

        # -- Supports* --
        for proto, dunder, ret in [
            (BuiltinProtocol.SUPPORTS_INT, "__int__", int_ty),
            (BuiltinProtocol.SUPPORTS_FLOAT, "__float__", any_ty),
            (BuiltinProtocol.SUPPORTS_COMPLEX, "__complex__", any_ty),
            (BuiltinProtocol.SUPPORTS_BYTES, "__bytes__", any_ty),
            (BuiltinProtocol.SUPPORTS_ABS, "__abs__", any_ty),
            (BuiltinProtocol.SUPPORTS_ROUND, "__round__", any_ty),
        ]:
            self.protocols[proto] = ProtocolDefinition(
                name=proto.name.replace("_", "").title(),
                required_methods={
                    dunder: FunctionSignature.of([self_param], ret),
                },
                is_runtime_checkable=True,
            )

        # -- Buffer --
        self.protocols[BuiltinProtocol.BUFFER] = ProtocolDefinition(
            name="Buffer",
            required_methods={
                "__buffer__": FunctionSignature.of(
                    [self_param, ("flags", int_ty)], any_ty
                ),
            },
            is_runtime_checkable=True,
        )


# ---------------------------------------------------------------------------
# ProtocolInference
# ---------------------------------------------------------------------------

class ProtocolInference:
    """Infer which protocols a variable satisfies from observed usage."""

    _USAGE_TO_PROTOCOL: Dict[str, Set[BuiltinProtocol]] = {
        "__len__": {BuiltinProtocol.SIZED},
        "__iter__": {BuiltinProtocol.ITERABLE},
        "__next__": {BuiltinProtocol.ITERATOR},
        "__contains__": {BuiltinProtocol.CONTAINER},
        "__hash__": {BuiltinProtocol.HASHABLE},
        "__reversed__": {BuiltinProtocol.REVERSIBLE},
        "__call__": {BuiltinProtocol.CALLABLE_},
        "__await__": {BuiltinProtocol.AWAITABLE},
        "__aiter__": {BuiltinProtocol.ASYNC_ITERABLE},
        "__anext__": {BuiltinProtocol.ASYNC_ITERATOR},
        "__enter__": {BuiltinProtocol.CONTEXT_MANAGER},
        "__exit__": {BuiltinProtocol.CONTEXT_MANAGER},
        "__aenter__": {BuiltinProtocol.ASYNC_CONTEXT_MANAGER},
        "__aexit__": {BuiltinProtocol.ASYNC_CONTEXT_MANAGER},
        "__int__": {BuiltinProtocol.SUPPORTS_INT},
        "__float__": {BuiltinProtocol.SUPPORTS_FLOAT},
        "__getitem__": {BuiltinProtocol.SEQUENCE, BuiltinProtocol.MAPPING},
        "__setitem__": {BuiltinProtocol.MUTABLE_SEQUENCE, BuiltinProtocol.MUTABLE_MAPPING},
        "__delitem__": {BuiltinProtocol.MUTABLE_SEQUENCE, BuiltinProtocol.MUTABLE_MAPPING},
        "keys": {BuiltinProtocol.MAPPING},
        "values": {BuiltinProtocol.MAPPING},
        "items": {BuiltinProtocol.MAPPING},
        "add": {BuiltinProtocol.MUTABLE_SET},
        "discard": {BuiltinProtocol.MUTABLE_SET},
        "insert": {BuiltinProtocol.MUTABLE_SEQUENCE},
    }

    # Friendly aliases for surface-level usage patterns
    _PATTERN_TO_USAGE: Dict[str, List[str]] = {
        "len()": ["__len__"],
        "for": ["__iter__"],
        "in": ["__contains__"],
        "[]": ["__getitem__"],
        "[]=": ["__setitem__"],
        "del[]": ["__delitem__"],
        "next()": ["__next__"],
        "hash()": ["__hash__"],
        "reversed()": ["__reversed__"],
        "call()": ["__call__"],
        "await": ["__await__"],
        "async for": ["__aiter__"],
        "with": ["__enter__", "__exit__"],
        "async with": ["__aenter__", "__aexit__"],
        "int()": ["__int__"],
        "float()": ["__float__"],
    }

    @classmethod
    def infer_from_usage(
        cls, var: str, usages: List[str]
    ) -> Set[BuiltinProtocol]:
        """Infer protocols from observed usage patterns on *var*."""
        resolved_usages: List[str] = []
        for u in usages:
            if u in cls._PATTERN_TO_USAGE:
                resolved_usages.extend(cls._PATTERN_TO_USAGE[u])
            else:
                resolved_usages.append(u)

        protocols: Set[BuiltinProtocol] = set()
        for usage in resolved_usages:
            if usage in cls._USAGE_TO_PROTOCOL:
                protocols |= cls._USAGE_TO_PROTOCOL[usage]
        return protocols

    @staticmethod
    def infer_from_annotation(annotation: str) -> Optional[ProtocolDefinition]:
        """Map a string annotation (e.g. ``"Iterable[int]"``) to a protocol."""
        _ANN_MAP: Dict[str, BuiltinProtocol] = {
            "Sized": BuiltinProtocol.SIZED,
            "Iterable": BuiltinProtocol.ITERABLE,
            "Iterator": BuiltinProtocol.ITERATOR,
            "Container": BuiltinProtocol.CONTAINER,
            "Hashable": BuiltinProtocol.HASHABLE,
            "Reversible": BuiltinProtocol.REVERSIBLE,
            "Sequence": BuiltinProtocol.SEQUENCE,
            "MutableSequence": BuiltinProtocol.MUTABLE_SEQUENCE,
            "Mapping": BuiltinProtocol.MAPPING,
            "MutableMapping": BuiltinProtocol.MUTABLE_MAPPING,
            "Set": BuiltinProtocol.SET_,
            "MutableSet": BuiltinProtocol.MUTABLE_SET,
            "Callable": BuiltinProtocol.CALLABLE_,
            "Awaitable": BuiltinProtocol.AWAITABLE,
            "AsyncIterable": BuiltinProtocol.ASYNC_ITERABLE,
            "AsyncIterator": BuiltinProtocol.ASYNC_ITERATOR,
            "ContextManager": BuiltinProtocol.CONTEXT_MANAGER,
            "AsyncContextManager": BuiltinProtocol.ASYNC_CONTEXT_MANAGER,
            "SupportsInt": BuiltinProtocol.SUPPORTS_INT,
            "SupportsFloat": BuiltinProtocol.SUPPORTS_FLOAT,
            "SupportsComplex": BuiltinProtocol.SUPPORTS_COMPLEX,
            "SupportsBytes": BuiltinProtocol.SUPPORTS_BYTES,
            "SupportsAbs": BuiltinProtocol.SUPPORTS_ABS,
            "SupportsRound": BuiltinProtocol.SUPPORTS_ROUND,
            "Buffer": BuiltinProtocol.BUFFER,
        }
        base_name = annotation.split("[")[0].strip()
        bp = _ANN_MAP.get(base_name)
        if bp is None:
            return None
        registry = BuiltinProtocolRegistry()
        return registry.get_protocol(bp)

    @staticmethod
    def infer_structural(
        methods_used: Set[str], attrs_used: Set[str]
    ) -> ProtocolType:
        """Build an ad-hoc structural ``ProtocolType`` from observed members."""
        method_types: Dict[str, FunctionPyType] = {}
        for m in methods_used:
            method_types[m] = FunctionPyType(
                params=[("self", AnyType())],
                return_type=AnyType(),
            )
        attr_types: Dict[str, PyType] = {}
        for a in attrs_used:
            attr_types[a] = AnyType()
        return ProtocolType(
            name="<inferred>",
            methods=method_types,
            attrs=attr_types,
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _type_name(ty: PyType) -> str:
    """Best-effort human-readable name for a PyType."""
    if hasattr(ty, "name"):
        return ty.name  # type: ignore[union-attr]
    return type(ty).__name__


def _is_subtype(sub: PyType, sup: PyType) -> bool:
    """Conservative structural subtype check."""
    if isinstance(sup, AnyType):
        return True
    if isinstance(sub, NeverType):
        return True
    if type(sub) is type(sup):
        return True
    if isinstance(sup, PyUnionType):
        members = getattr(sup, "members", [])
        return any(_is_subtype(sub, m) for m in members)
    return False


def _to_function_signature(method: Any) -> Optional[FunctionSignature]:
    """Try to convert a method descriptor into a ``FunctionSignature``."""
    if isinstance(method, FunctionSignature):
        return method
    if isinstance(method, FunctionPyType):
        params = [(n, t) for n, t in getattr(method, "params", [])]
        return FunctionSignature.of(params, getattr(method, "return_type", AnyType()))
    return None


def _join_ref(a: PyRefinementType, b: PyRefinementType) -> PyRefinementType:
    """Lattice join for refinement types (upper bound / less precise)."""
    a_base = getattr(a, "base", None)
    b_base = getattr(b, "base", None)
    if a_base is not None and b_base is not None:
        if isinstance(a_base, AnyType) or isinstance(b_base, AnyType):
            return PyRefinementType(base=AnyType())
        if type(a_base) is type(b_base):
            return a
    return PyRefinementType(base=AnyType())
