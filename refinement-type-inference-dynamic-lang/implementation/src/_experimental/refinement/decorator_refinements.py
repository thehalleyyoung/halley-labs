"""Decorator refinement analysis for Python functions and classes.

Models how decorators transform function and class types, synthesizing
refined signatures for property, staticmethod, classmethod, dataclass,
contextmanager, lru_cache, overload, and other standard decorators.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from src.refinement_lattice import (
    ANY_TYPE,
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    NEVER_TYPE,
    NONE_TYPE,
    STR_TYPE,
    BaseTypeKind,
    BaseTypeR,
    Pred,
    PredOp,
    RefType,
)


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassContext:
    """Context for a method defined inside a class body."""
    class_name: str
    bases: Tuple[str, ...] = ()
    is_method: bool = True


@dataclass(frozen=True)
class FieldInfo:
    """A single field in a dataclass or similar structured class."""
    name: str
    type: BaseTypeR = ANY_TYPE
    default: Optional[ast.expr] = None
    has_default: bool = False
    init: bool = True
    repr_: bool = True
    compare: bool = True
    hash_: Optional[bool] = None
    kw_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FunctionSignature:
    """Extracted or synthesized function signature."""
    name: str
    param_names: Tuple[str, ...] = ()
    param_types: Tuple[RefType, ...] = ()
    return_type: RefType = field(default_factory=lambda: RefType.trivial(ANY_TYPE))
    decorators: Tuple[str, ...] = ()
    is_async: bool = False
    is_generator: bool = False
    has_self: bool = False
    has_cls: bool = False
    vararg: Optional[str] = None
    kwarg: Optional[str] = None
    defaults_count: int = 0


@dataclass(frozen=True)
class FunctionRefinement:
    """Refined function type with pre/post conditions and effects."""
    signature: FunctionSignature
    preconditions: Tuple[Pred, ...] = ()
    postconditions: Tuple[Pred, ...] = ()
    effects: FrozenSet[str] = frozenset()
    is_abstract: bool = False
    is_override: bool = False
    preserves_signature: bool = False
    cache_info: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PropertyType:
    """Refined type for a property descriptor."""
    name: str
    getter_type: Optional[RefType] = None
    setter_type: Optional[RefType] = None
    has_deleter: bool = False
    cached: bool = False
    owner_class: Optional[str] = None


@dataclass(frozen=True)
class ContextManagerType:
    """Refined type for a context manager produced by @contextmanager."""
    enter_type: RefType = field(default_factory=lambda: RefType.trivial(ANY_TYPE))
    exit_type: RefType = field(default_factory=lambda: RefType.trivial(NONE_TYPE))
    yield_type: RefType = field(default_factory=lambda: RefType.trivial(ANY_TYPE))
    manages_resources: bool = True
    is_async: bool = False
    func_name: str = ""


@dataclass(frozen=True)
class OverloadedFunctionType:
    """Multiple overloaded signatures for a single function name."""
    name: str
    signatures: Tuple[FunctionSignature, ...] = ()
    implementation: Optional[FunctionSignature] = None

    def dispatch(self, arg_types: Sequence[BaseTypeR]) -> Optional[FunctionSignature]:
        """Select the first matching overload for the given argument types."""
        for sig in self.signatures:
            if self._matches(sig, arg_types):
                return sig
        return self.implementation

    @staticmethod
    def _matches(sig: FunctionSignature, arg_types: Sequence[BaseTypeR]) -> bool:
        params = sig.param_types
        offset = 1 if sig.has_self or sig.has_cls else 0
        effective = params[offset:]
        if len(arg_types) > len(effective):
            return sig.vararg is not None
        for arg_t, param_rt in zip(arg_types, effective):
            if not arg_t.is_subtype_of(param_rt.base):
                return False
        required = len(effective) - sig.defaults_count
        if len(arg_types) < required:
            return False
        return True


@dataclass(frozen=True)
class ClassRefinement:
    """Refinement info for a class transformed by a decorator (e.g. @dataclass)."""
    class_name: str
    fields: Tuple[FieldInfo, ...] = ()
    synthesized_methods: Tuple[FunctionSignature, ...] = ()
    frozen: bool = False
    slots: bool = False
    eq: bool = True
    order: bool = False
    hash_: Optional[bool] = None
    match_args: bool = True
    kw_only: bool = False


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

_ANNOTATION_MAP: Dict[str, BaseTypeR] = {
    "int": INT_TYPE,
    "float": FLOAT_TYPE,
    "str": STR_TYPE,
    "bool": BOOL_TYPE,
    "None": NONE_TYPE,
    "list": BaseTypeR(BaseTypeKind.LIST),
    "dict": BaseTypeR(BaseTypeKind.DICT),
    "set": BaseTypeR(BaseTypeKind.SET),
    "tuple": BaseTypeR(BaseTypeKind.TUPLE),
    "object": BaseTypeR(BaseTypeKind.OBJECT),
    "Any": ANY_TYPE,
}


def _resolve_annotation(node: Optional[ast.expr]) -> BaseTypeR:
    """Best-effort resolution of a type annotation AST node."""
    if node is None:
        return ANY_TYPE
    if isinstance(node, ast.Constant) and node.value is None:
        return NONE_TYPE
    if isinstance(node, ast.Name):
        return _ANNOTATION_MAP.get(node.id, ANY_TYPE)
    if isinstance(node, ast.Attribute):
        return ANY_TYPE
    if isinstance(node, ast.Subscript):
        base = _resolve_annotation(node.value) if isinstance(node.value, ast.Name) else ANY_TYPE
        return base
    return ANY_TYPE


def _has_yield(func: ast.FunctionDef) -> bool:
    """Return True if *func* contains a yield / yield from statement."""
    for node in ast.walk(func):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def _extract_yield_type(func: ast.FunctionDef) -> BaseTypeR:
    """Extract the yielded type from a generator function."""
    for node in ast.walk(func):
        if isinstance(node, ast.Yield) and node.value is not None:
            if isinstance(node.value, ast.Constant):
                val = node.value.value
                if isinstance(val, int):
                    return INT_TYPE
                if isinstance(val, float):
                    return FLOAT_TYPE
                if isinstance(val, str):
                    return STR_TYPE
                if isinstance(val, bool):
                    return BOOL_TYPE
                if val is None:
                    return NONE_TYPE
            return ANY_TYPE
    return NONE_TYPE


# ---------------------------------------------------------------------------
# Known-decorator registry
# ---------------------------------------------------------------------------

KNOWN_DECORATORS: Dict[str, str] = {
    "property": "model_property",
    "staticmethod": "model_staticmethod",
    "classmethod": "model_classmethod",
    "abstractmethod": "model_abstractmethod",
    "abc.abstractmethod": "model_abstractmethod",
    "contextmanager": "model_contextmanager",
    "contextlib.contextmanager": "model_contextmanager",
    "cached_property": "model_cached_property",
    "functools.cached_property": "model_cached_property",
    "dataclass": "model_dataclass",
    "dataclasses.dataclass": "model_dataclass",
    "overload": "model_overload",
    "typing.overload": "model_overload",
    "lru_cache": "model_lru_cache",
    "functools.lru_cache": "model_lru_cache",
    "wraps": "model_wraps",
    "functools.wraps": "model_wraps",
}


# ---------------------------------------------------------------------------
# DecoratorAnalyzer
# ---------------------------------------------------------------------------

class DecoratorAnalyzer:
    """Analyse decorator expressions and produce refined function/class types.

    The analyzer resolves each decorator to a known handler, extracts keyword
    arguments when the decorator is called, and delegates to a specific
    ``model_*`` method that builds the appropriate refinement type.
    """

    def __init__(self) -> None:
        self._overload_groups: Dict[str, List[ast.FunctionDef]] = {}

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def analyze_decorator(
        self,
        decorator: ast.expr,
        func: ast.FunctionDef,
        class_ctx: Optional[ClassContext] = None,
    ) -> FunctionRefinement:
        """Analyse a single decorator applied to *func* and return a refinement."""
        name = self._resolve_decorator_name(decorator)
        if name is None:
            return self._default_refinement(func, class_ctx)

        handler_name = KNOWN_DECORATORS.get(name)
        if handler_name is None:
            return self._default_refinement(func, class_ctx)

        args = self._extract_decorator_args(decorator)

        if handler_name == "model_property":
            kind = args.get("kind", "getter")
            prop = self.model_property(func, kind=kind)
            sig = self._extract_function_signature(func)
            return FunctionRefinement(
                signature=sig,
                postconditions=(
                    Pred(PredOp.IS_NOT_NONE, ("return",)),
                ) if prop.getter_type is not None else (),
            )

        if handler_name == "model_staticmethod":
            return self.model_staticmethod(func)

        if handler_name == "model_classmethod":
            return self.model_classmethod(func)

        if handler_name == "model_abstractmethod":
            return self.model_abstractmethod(func)

        if handler_name == "model_contextmanager":
            cm = self.model_contextmanager(func)
            sig = self._extract_function_signature(func)
            return FunctionRefinement(
                signature=sig,
                effects=frozenset({"resource_management"}),
            )

        if handler_name == "model_cached_property":
            prop = self.model_cached_property(func)
            sig = self._extract_function_signature(func)
            return FunctionRefinement(signature=sig)

        if handler_name == "model_lru_cache":
            maxsize = args.get("maxsize", 128)
            return self.model_lru_cache(func, maxsize=maxsize)

        if handler_name == "model_overload":
            self._overload_groups.setdefault(func.name, []).append(func)
            sig = self._extract_function_signature(func)
            return FunctionRefinement(signature=sig)

        if handler_name == "model_wraps":
            sig = self._extract_function_signature(func)
            return FunctionRefinement(signature=sig, preserves_signature=True)

        return self._default_refinement(func, class_ctx)

    def analyze_stacked_decorators(
        self,
        decorators: List[ast.expr],
        func: ast.FunctionDef,
    ) -> FunctionRefinement:
        """Analyse a stack of decorators applied bottom-up (innermost first).

        Decorators are applied from bottom to top in Python, so we iterate
        in reverse order.  Preconditions and postconditions are accumulated;
        effects are unioned; the signature may be rewritten by each layer.
        """
        preconditions: List[Pred] = []
        postconditions: List[Pred] = []
        effects: Set[str] = set()
        is_abstract = False
        preserves_sig = False
        cache_info: Optional[Dict[str, Any]] = None

        sig = self._extract_function_signature(func)

        for dec in reversed(decorators):
            ref = self.analyze_decorator(dec, func)
            sig = ref.signature
            preconditions.extend(ref.preconditions)
            postconditions.extend(ref.postconditions)
            effects.update(ref.effects)
            if ref.is_abstract:
                is_abstract = True
            if ref.preserves_signature:
                preserves_sig = True
            if ref.cache_info is not None:
                cache_info = ref.cache_info

        return FunctionRefinement(
            signature=sig,
            preconditions=tuple(preconditions),
            postconditions=tuple(postconditions),
            effects=frozenset(effects),
            is_abstract=is_abstract,
            preserves_signature=preserves_sig,
            cache_info=cache_info,
        )

    # ------------------------------------------------------------------
    # model_* handlers
    # ------------------------------------------------------------------

    def model_property(self, func: ast.FunctionDef, kind: str = "getter") -> PropertyType:
        """Model a ``@property`` (getter), ``.setter``, or ``.deleter``."""
        ret_ann = _resolve_annotation(func.returns)
        getter_type: Optional[RefType] = None
        setter_type: Optional[RefType] = None
        has_deleter = False

        if kind == "getter":
            getter_type = RefType.trivial(ret_ann)
        elif kind == "setter":
            # setter's second param determines the accepted type
            if len(func.args.args) >= 2:
                ann_node = func.args.args[1].annotation
                setter_type = RefType.trivial(_resolve_annotation(ann_node))
            else:
                setter_type = RefType.trivial(ANY_TYPE)
        elif kind == "deleter":
            has_deleter = True

        return PropertyType(
            name=func.name,
            getter_type=getter_type,
            setter_type=setter_type,
            has_deleter=has_deleter,
            cached=False,
        )

    def model_staticmethod(self, func: ast.FunctionDef) -> FunctionRefinement:
        """Model ``@staticmethod`` – no implicit ``self`` parameter."""
        sig = self._extract_function_signature(func)
        # Strip has_self / has_cls; params stay as-is since there is no self.
        sig = FunctionSignature(
            name=sig.name,
            param_names=sig.param_names,
            param_types=sig.param_types,
            return_type=sig.return_type,
            decorators=("staticmethod",) + sig.decorators,
            is_async=sig.is_async,
            is_generator=sig.is_generator,
            has_self=False,
            has_cls=False,
            vararg=sig.vararg,
            kwarg=sig.kwarg,
            defaults_count=sig.defaults_count,
        )
        return FunctionRefinement(signature=sig)

    def model_classmethod(self, func: ast.FunctionDef) -> FunctionRefinement:
        """Model ``@classmethod`` – first arg receives the class object."""
        sig = self._extract_function_signature(func)
        # Mark has_cls; first parameter type is ``type[Self]``.
        first_type = RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))
        param_types = (first_type,) + sig.param_types[1:] if sig.param_types else (first_type,)
        sig = FunctionSignature(
            name=sig.name,
            param_names=sig.param_names,
            param_types=param_types,
            return_type=sig.return_type,
            decorators=("classmethod",) + sig.decorators,
            is_async=sig.is_async,
            is_generator=sig.is_generator,
            has_self=False,
            has_cls=True,
            vararg=sig.vararg,
            kwarg=sig.kwarg,
            defaults_count=sig.defaults_count,
        )
        pre = (
            Pred(PredOp.IS_NOT_NONE, ("cls",)),
        )
        return FunctionRefinement(signature=sig, preconditions=pre)

    def model_abstractmethod(self, func: ast.FunctionDef) -> FunctionRefinement:
        """Model ``@abstractmethod`` – must be overridden by subclasses."""
        sig = self._extract_function_signature(func)
        sig = FunctionSignature(
            name=sig.name,
            param_names=sig.param_names,
            param_types=sig.param_types,
            return_type=sig.return_type,
            decorators=("abstractmethod",) + sig.decorators,
            is_async=sig.is_async,
            is_generator=sig.is_generator,
            has_self=sig.has_self,
            has_cls=sig.has_cls,
            vararg=sig.vararg,
            kwarg=sig.kwarg,
            defaults_count=sig.defaults_count,
        )
        return FunctionRefinement(
            signature=sig,
            is_abstract=True,
            postconditions=(Pred(PredOp.TRUE),),
        )

    def model_contextmanager(self, gen_func: ast.FunctionDef) -> ContextManagerType:
        """Model ``@contextmanager`` – yield becomes ``__enter__`` return.

        Code after the ``yield`` is the ``__exit__`` logic.  The yielded
        value determines the ``__enter__`` return type.
        """
        is_gen = _has_yield(gen_func)
        yield_base = _extract_yield_type(gen_func) if is_gen else NONE_TYPE
        enter_type = RefType.trivial(yield_base)
        exit_type = RefType.trivial(NONE_TYPE)
        is_async = isinstance(gen_func, ast.AsyncFunctionDef)

        return ContextManagerType(
            enter_type=enter_type,
            exit_type=exit_type,
            yield_type=RefType.trivial(yield_base),
            manages_resources=True,
            is_async=is_async,
            func_name=gen_func.name,
        )

    def model_cached_property(self, prop: ast.FunctionDef) -> PropertyType:
        """Model ``@cached_property`` – computed once, stored as attribute."""
        ret_ann = _resolve_annotation(prop.returns)
        return PropertyType(
            name=prop.name,
            getter_type=RefType.trivial(ret_ann),
            setter_type=None,
            has_deleter=False,
            cached=True,
        )

    def model_dataclass(
        self,
        cls: ast.ClassDef,
        frozen: bool = False,
        slots: bool = False,
        eq: bool = True,
        order: bool = False,
    ) -> ClassRefinement:
        """Model ``@dataclass`` – synthesize ``__init__``, ``__eq__``, etc."""
        fields = self._extract_dataclass_fields(cls)
        methods: List[FunctionSignature] = []

        methods.append(self._synthesize_init(cls, fields))

        methods.append(self._synthesize_repr(cls, fields))

        if eq:
            methods.append(self._synthesize_eq(cls, fields))

        hash_val: Optional[bool] = None
        if eq and not frozen:
            hash_val = False
        elif frozen:
            hash_val = True
            methods.append(self._synthesize_hash(cls, fields))

        if order:
            methods.extend(self._synthesize_order_methods(cls, fields))

        return ClassRefinement(
            class_name=cls.name,
            fields=tuple(fields),
            synthesized_methods=tuple(methods),
            frozen=frozen,
            slots=slots,
            eq=eq,
            order=order,
            hash_=hash_val,
        )

    def model_overload(
        self, overloads: List[ast.FunctionDef],
    ) -> OverloadedFunctionType:
        """Model ``@overload`` – combine overloaded signatures."""
        if not overloads:
            return OverloadedFunctionType(name="<unknown>")
        sigs: List[FunctionSignature] = []
        for func in overloads:
            sigs.append(self._extract_function_signature(func))
        name = overloads[0].name
        return OverloadedFunctionType(
            name=name,
            signatures=tuple(sigs),
            implementation=None,
        )

    def model_lru_cache(
        self, func: ast.FunctionDef, maxsize: Optional[int] = 128,
    ) -> FunctionRefinement:
        """Model ``@lru_cache`` – memoize with optional max size."""
        sig = self._extract_function_signature(func)

        # All parameters must be hashable (frozen / immutable).
        preconds: List[Pred] = []
        offset = 1 if sig.has_self or sig.has_cls else 0
        for pname in sig.param_names[offset:]:
            preconds.append(Pred(PredOp.HASATTR, (pname, "__hash__")))

        cache = {"maxsize": maxsize, "typed": False}
        return FunctionRefinement(
            signature=sig,
            preconditions=tuple(preconds),
            effects=frozenset({"memoization"}),
            cache_info=cache,
        )

    def model_wraps(
        self,
        wrapper: ast.FunctionDef,
        wrapped: ast.FunctionDef,
    ) -> FunctionRefinement:
        """Model ``@wraps(wrapped)`` – wrapper adopts wrapped's signature."""
        wrapped_sig = self._extract_function_signature(wrapped)
        wrapper_sig = self._extract_function_signature(wrapper)

        combined = FunctionSignature(
            name=wrapped_sig.name,
            param_names=wrapped_sig.param_names,
            param_types=wrapped_sig.param_types,
            return_type=wrapper_sig.return_type,
            decorators=wrapper_sig.decorators,
            is_async=wrapper_sig.is_async,
            is_generator=wrapper_sig.is_generator,
            has_self=wrapped_sig.has_self,
            has_cls=wrapped_sig.has_cls,
            vararg=wrapped_sig.vararg,
            kwarg=wrapped_sig.kwarg,
            defaults_count=wrapped_sig.defaults_count,
        )
        return FunctionRefinement(
            signature=combined,
            preserves_signature=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers – decorator resolution
    # ------------------------------------------------------------------

    def _resolve_decorator_name(self, decorator: ast.expr) -> Optional[str]:
        """Resolve a decorator AST node to a dotted name string.

        Handles bare names (``@property``), attribute access
        (``@abc.abstractmethod``), and calls (``@dataclass(frozen=True)``).
        For ``.setter`` / ``.deleter`` attribute chains on property names
        we normalise to ``"property"``.
        """
        if isinstance(decorator, ast.Name):
            return decorator.id

        if isinstance(decorator, ast.Attribute):
            # e.g. @abc.abstractmethod  or  @prop.setter
            if isinstance(decorator.value, ast.Name):
                prefix = decorator.value.id
                suffix = decorator.attr
                if suffix in ("setter", "deleter", "getter"):
                    return "property"
                return f"{prefix}.{suffix}"
            return None

        if isinstance(decorator, ast.Call):
            return self._resolve_decorator_name(decorator.func)

        return None

    def _extract_decorator_args(self, decorator: ast.expr) -> Dict[str, Any]:
        """Extract keyword arguments from a decorator call.

        For ``@dataclass(frozen=True, order=False)`` returns
        ``{"frozen": True, "order": False}``.  Non-call decorators yield an
        empty dict.
        """
        result: Dict[str, Any] = {}
        if not isinstance(decorator, ast.Call):
            # Check for .setter/.deleter attribute access
            if isinstance(decorator, ast.Attribute):
                if decorator.attr in ("setter", "deleter", "getter"):
                    result["kind"] = decorator.attr
            return result

        for kw in decorator.keywords:
            if kw.arg is None:
                continue
            val = self._eval_constant(kw.value)
            result[kw.arg] = val

        # Positional args for decorators like @lru_cache(maxsize)
        if decorator.args and not result:
            first = self._eval_constant(decorator.args[0])
            if first is not None:
                # Heuristic: first positional is maxsize for lru_cache
                name = self._resolve_decorator_name(decorator)
                if name and "lru_cache" in name:
                    result["maxsize"] = first

        return result

    # ------------------------------------------------------------------
    # Internal helpers – signature extraction
    # ------------------------------------------------------------------

    def _extract_function_signature(self, func: ast.FunctionDef) -> FunctionSignature:
        """Build a ``FunctionSignature`` from an AST function definition."""
        args = func.args
        param_names: List[str] = []
        param_types: List[RefType] = []

        for arg in args.args:
            param_names.append(arg.arg)
            base = _resolve_annotation(arg.annotation)
            param_types.append(RefType.trivial(base))

        has_self = bool(param_names) and param_names[0] == "self"
        has_cls = bool(param_names) and param_names[0] == "cls" and not has_self

        ret_base = _resolve_annotation(func.returns)
        ret_type = RefType.trivial(ret_base)

        is_gen = _has_yield(func)
        is_async = isinstance(func, ast.AsyncFunctionDef)

        vararg = args.vararg.arg if args.vararg else None
        kwarg = args.kwarg.arg if args.kwarg else None

        num_defaults = len(args.defaults)

        dec_names: List[str] = []
        for d in func.decorator_list:
            n = self._resolve_decorator_name(d)
            if n is not None:
                dec_names.append(n)

        return FunctionSignature(
            name=func.name,
            param_names=tuple(param_names),
            param_types=tuple(param_types),
            return_type=ret_type,
            decorators=tuple(dec_names),
            is_async=is_async,
            is_generator=is_gen,
            has_self=has_self,
            has_cls=has_cls,
            vararg=vararg,
            kwarg=kwarg,
            defaults_count=num_defaults,
        )

    # ------------------------------------------------------------------
    # Internal helpers – dataclass synthesis
    # ------------------------------------------------------------------

    def _extract_dataclass_fields(self, cls: ast.ClassDef) -> List[FieldInfo]:
        """Walk class body and collect field definitions."""
        fields: List[FieldInfo] = []
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                ftype = _resolve_annotation(stmt.annotation)
                has_default = stmt.value is not None
                default = stmt.value if has_default else None
                meta: Dict[str, Any] = {}

                # Detect field() calls
                init_flag = True
                repr_flag = True
                compare_flag = True
                hash_flag: Optional[bool] = None
                kw_only = False

                if isinstance(stmt.value, ast.Call):
                    func_node = stmt.value.func
                    if isinstance(func_node, ast.Name) and func_node.id == "field":
                        for kw in stmt.value.keywords:
                            if kw.arg == "init":
                                init_flag = self._eval_constant(kw.value) is not False
                            elif kw.arg == "repr":
                                repr_flag = self._eval_constant(kw.value) is not False
                            elif kw.arg == "compare":
                                compare_flag = self._eval_constant(kw.value) is not False
                            elif kw.arg == "hash":
                                hash_flag = self._eval_constant(kw.value)
                            elif kw.arg == "kw_only":
                                kw_only = self._eval_constant(kw.value) is True
                            elif kw.arg == "default":
                                has_default = True
                            elif kw.arg == "default_factory":
                                has_default = True
                            elif kw.arg == "metadata":
                                meta = {"has_metadata": True}

                fields.append(FieldInfo(
                    name=name,
                    type=ftype,
                    default=default,
                    has_default=has_default,
                    init=init_flag,
                    repr_=repr_flag,
                    compare=compare_flag,
                    hash_=hash_flag,
                    kw_only=kw_only,
                    metadata=meta,
                ))
        return fields

    def _synthesize_init(
        self, cls: ast.ClassDef, fields: List[FieldInfo],
    ) -> FunctionSignature:
        """Synthesize ``__init__`` for a dataclass."""
        param_names: List[str] = ["self"]
        param_types: List[RefType] = [RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))]
        defaults_count = 0

        for f in fields:
            if not f.init:
                continue
            param_names.append(f.name)
            param_types.append(RefType.trivial(f.type))
            if f.has_default:
                defaults_count += 1

        return FunctionSignature(
            name="__init__",
            param_names=tuple(param_names),
            param_types=tuple(param_types),
            return_type=RefType.trivial(NONE_TYPE),
            has_self=True,
            defaults_count=defaults_count,
        )

    def _synthesize_eq(
        self, cls: ast.ClassDef, fields: List[FieldInfo],
    ) -> FunctionSignature:
        """Synthesize ``__eq__`` comparing all fields with ``compare=True``."""
        self_type = RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))
        other_type = RefType.trivial(ANY_TYPE)
        return FunctionSignature(
            name="__eq__",
            param_names=("self", "other"),
            param_types=(self_type, other_type),
            return_type=RefType.trivial(BOOL_TYPE),
            has_self=True,
        )

    def _synthesize_hash(
        self, cls: ast.ClassDef, fields: List[FieldInfo],
    ) -> FunctionSignature:
        """Synthesize ``__hash__`` using fields with ``hash=True`` or ``compare=True``."""
        self_type = RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))
        return FunctionSignature(
            name="__hash__",
            param_names=("self",),
            param_types=(self_type,),
            return_type=RefType.trivial(INT_TYPE),
            has_self=True,
        )

    def _synthesize_repr(
        self, cls: ast.ClassDef, fields: List[FieldInfo],
    ) -> FunctionSignature:
        """Synthesize ``__repr__`` returning a string representation."""
        self_type = RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))
        return FunctionSignature(
            name="__repr__",
            param_names=("self",),
            param_types=(self_type,),
            return_type=RefType.trivial(STR_TYPE),
            has_self=True,
        )

    def _synthesize_order_methods(
        self, cls: ast.ClassDef, fields: List[FieldInfo],
    ) -> List[FunctionSignature]:
        """Synthesize ``__lt__``, ``__le__``, ``__gt__``, ``__ge__``."""
        self_type = RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))
        other_type = RefType.trivial(BaseTypeR(BaseTypeKind.OBJECT))
        methods: List[FunctionSignature] = []
        for dunder in ("__lt__", "__le__", "__gt__", "__ge__"):
            methods.append(FunctionSignature(
                name=dunder,
                param_names=("self", "other"),
                param_types=(self_type, other_type),
                return_type=RefType.trivial(BOOL_TYPE),
                has_self=True,
            ))
        return methods

    # ------------------------------------------------------------------
    # Internal helpers – misc
    # ------------------------------------------------------------------

    def _default_refinement(
        self,
        func: ast.FunctionDef,
        class_ctx: Optional[ClassContext] = None,
    ) -> FunctionRefinement:
        """Fallback refinement for unrecognised decorators."""
        sig = self._extract_function_signature(func)
        preconds: Tuple[Pred, ...] = ()
        if class_ctx is not None and class_ctx.is_method and sig.has_self:
            preconds = (Pred(PredOp.IS_NOT_NONE, ("self",)),)
        return FunctionRefinement(signature=sig, preconditions=preconds)

    @staticmethod
    def _eval_constant(node: ast.expr) -> Any:
        """Attempt to evaluate an AST node as a Python constant."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            if node.id == "None":
                return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = DecoratorAnalyzer._eval_constant(node.operand)
            if isinstance(inner, (int, float)):
                return -inner
        return None

    def flush_overloads(self) -> Dict[str, OverloadedFunctionType]:
        """Consume accumulated ``@overload`` groups and return resolved types."""
        result: Dict[str, OverloadedFunctionType] = {}
        for name, funcs in self._overload_groups.items():
            result[name] = self.model_overload(funcs)
        self._overload_groups.clear()
        return result
