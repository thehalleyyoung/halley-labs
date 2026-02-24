"""
Mutation semantics for heap-aware refinement type inference.

Models how Python mutations (attribute sets, item sets, method calls on
collections) affect refinement types.  Given a mutation operation this module
computes:

  1. Which heap locations are modified (MutationEffect).
  2. Which existing refinement predicates are invalidated
     (RefinementInvalidation).
  3. Summary frames for entire functions (ModificationFrame) that describe
     what a call may read/write.
  4. Surviving refinements after a sequence of mutations.
  5. Weakest-precondition reasoning over single mutations.

Designed to work alongside:
  - class_hierarchy.ClassHierarchyAnalyzer / ClassInfo / MethodInfo
  - descriptor_protocol.DescriptorAnalyzer / DescriptorInfo /
    AttributeResolution
"""

from __future__ import annotations

import ast
import copy
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
    Union,
)

from .class_hierarchy import ClassHierarchyAnalyzer, ClassInfo, MethodInfo
from .descriptor_protocol import DescriptorAnalyzer, DescriptorInfo, AttributeResolution


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModificationFrame:
    """Summary of what a function body may read and write."""

    modified_params: Set[str] = field(default_factory=set)
    modified_attrs: Dict[str, Set[str]] = field(default_factory=dict)
    modified_globals: Set[str] = field(default_factory=set)
    may_modify_globals: bool = False
    may_raise: Set[str] = field(default_factory=set)
    pure: bool = True
    reads_params: Set[str] = field(default_factory=set)
    reads_globals: Set[str] = field(default_factory=set)

    # ----- helpers ----------------------------------------------------------

    def mark_param_modified(self, param: str) -> None:
        self.modified_params.add(param)
        self.pure = False

    def mark_attr_modified(self, param: str, attr: str) -> None:
        self.modified_attrs.setdefault(param, set()).add(attr)
        self.modified_params.add(param)
        self.pure = False

    def mark_global_modified(self, name: str) -> None:
        self.modified_globals.add(name)
        self.may_modify_globals = True
        self.pure = False

    def merge(self, other: ModificationFrame) -> None:
        """Merge another frame into this one (union semantics)."""
        self.modified_params |= other.modified_params
        for param, attrs in other.modified_attrs.items():
            self.modified_attrs.setdefault(param, set()).update(attrs)
        self.modified_globals |= other.modified_globals
        self.may_modify_globals = self.may_modify_globals or other.may_modify_globals
        self.may_raise |= other.may_raise
        self.pure = self.pure and other.pure
        self.reads_params |= other.reads_params
        self.reads_globals |= other.reads_globals


@dataclass
class MutationEffect:
    """A single heap-mutation operation."""

    target: str
    kind: str  # setattr | delattr | setitem | delitem | append | extend |
    # insert | pop | remove | clear | sort | reverse | update |
    # add | discard
    attr_or_key: Optional[str] = None
    value_type: Optional[str] = None

    _VALID_KINDS: frozenset = frozenset({
        "setattr", "delattr", "setitem", "delitem",
        "append", "extend", "insert", "pop", "remove",
        "clear", "sort", "reverse", "update", "add", "discard",
    })

    def __post_init__(self) -> None:
        if self.kind not in self._VALID_KINDS:
            raise ValueError(f"Unknown mutation kind: {self.kind!r}")


@dataclass
class RefinementInvalidation:
    """Records that refinement predicates on *variable* are no longer valid."""

    variable: str
    invalidated_attrs: Set[str] = field(default_factory=set)
    reason: str = ""


@dataclass
class ImmutabilityInfo:
    """Whether a type is (deeply) immutable."""

    is_immutable: bool
    reason: str = ""
    mutable_attrs: Set[str] = field(default_factory=set)


@dataclass
class CollectionMutationResult:
    """Result of analysing a single collection-mutation method call."""

    new_length_relation: Optional[str] = None  # e.g. "old_len + 1"
    element_type_change: Optional[str] = None
    may_raise: bool = False
    exception_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Immutable-type registry
# ---------------------------------------------------------------------------

_BUILTIN_IMMUTABLE_TYPES: FrozenSet[str] = frozenset({
    "int", "float", "complex", "str", "bytes",
    "tuple", "frozenset", "range", "bool", "NoneType", "type",
})

# Methods on builtin mutable collections that we know about.
_LIST_MUTATION_METHODS: FrozenSet[str] = frozenset({
    "append", "extend", "insert", "pop", "remove",
    "clear", "sort", "reverse", "__iadd__", "__setitem__", "__delitem__",
})

_DICT_MUTATION_METHODS: FrozenSet[str] = frozenset({
    "__setitem__", "__delitem__", "update", "pop", "popitem",
    "setdefault", "clear",
})

_SET_MUTATION_METHODS: FrozenSet[str] = frozenset({
    "add", "discard", "remove", "pop", "clear",
    "update", "intersection_update", "difference_update",
    "symmetric_difference_update",
    "__ior__", "__iand__", "__isub__", "__ixor__",
})


# ---------------------------------------------------------------------------
# MutationSemantics
# ---------------------------------------------------------------------------

class MutationSemantics:
    """Heap-aware mutation analysis for refinement type inference."""

    def __init__(
        self,
        hierarchy: ClassHierarchyAnalyzer,
        descriptors: DescriptorAnalyzer,
    ) -> None:
        self.hierarchy = hierarchy
        self.descriptors = descriptors

        self._immutable_types: Set[str] = self._init_immutable_types()
        self._list_mutations: Dict[str, Callable[..., CollectionMutationResult]] = (
            self._init_list_mutations()
        )
        self._dict_mutations: Dict[str, Callable[..., CollectionMutationResult]] = (
            self._init_dict_mutations()
        )
        self._set_mutations: Dict[str, Callable[..., CollectionMutationResult]] = (
            self._init_set_mutations()
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_immutable_types(self) -> Set[str]:
        return set(_BUILTIN_IMMUTABLE_TYPES)

    def _init_list_mutations(self) -> Dict[str, Callable[..., CollectionMutationResult]]:
        return {
            "append": self._list_append,
            "extend": self._list_extend,
            "insert": self._list_insert,
            "pop": self._list_pop,
            "remove": self._list_remove,
            "clear": self._list_clear,
            "sort": self._list_sort,
            "reverse": self._list_reverse,
            "__iadd__": self._list_extend,
            "__setitem__": self._list_setitem,
            "__delitem__": self._list_delitem,
        }

    def _init_dict_mutations(self) -> Dict[str, Callable[..., CollectionMutationResult]]:
        return {
            "__setitem__": self._dict_setitem,
            "__delitem__": self._dict_delitem,
            "update": self._dict_update,
            "pop": self._dict_pop,
            "popitem": self._dict_popitem,
            "setdefault": self._dict_setdefault,
            "clear": self._dict_clear,
        }

    def _init_set_mutations(self) -> Dict[str, Callable[..., CollectionMutationResult]]:
        return {
            "add": self._set_add,
            "discard": self._set_discard,
            "remove": self._set_remove,
            "pop": self._set_pop,
            "clear": self._set_clear,
            "update": self._set_update,
            "intersection_update": self._set_intersection_update,
            "difference_update": self._set_difference_update,
            "symmetric_difference_update": self._set_symmetric_difference_update,
            "__ior__": self._set_update,
            "__iand__": self._set_intersection_update,
            "__isub__": self._set_difference_update,
            "__ixor__": self._set_symmetric_difference_update,
        }

    # ------------------------------------------------------------------
    # List mutation handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_append(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[0] if arg_types else None
        return CollectionMutationResult(
            new_length_relation="old_len + 1",
            element_type_change=elem_change,
        )

    @staticmethod
    def _list_extend(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[0] if arg_types else None
        return CollectionMutationResult(
            new_length_relation="old_len + other_len",
            element_type_change=elem_change,
        )

    @staticmethod
    def _list_insert(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[1] if arg_types and len(arg_types) > 1 else None
        return CollectionMutationResult(
            new_length_relation="old_len + 1",
            element_type_change=elem_change,
        )

    @staticmethod
    def _list_pop(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="IndexError",
        )

    @staticmethod
    def _list_remove(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="ValueError",
        )

    @staticmethod
    def _list_clear(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(new_length_relation="0")

    @staticmethod
    def _list_sort(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(new_length_relation="old_len")

    @staticmethod
    def _list_reverse(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(new_length_relation="old_len")

    @staticmethod
    def _list_setitem(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[1] if arg_types and len(arg_types) > 1 else None
        return CollectionMutationResult(
            new_length_relation="old_len",
            element_type_change=elem_change,
            may_raise=True,
            exception_type="IndexError",
        )

    @staticmethod
    def _list_delitem(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="IndexError",
        )

    # ------------------------------------------------------------------
    # Dict mutation handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_setitem(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[1] if arg_types and len(arg_types) > 1 else None
        return CollectionMutationResult(
            new_length_relation="old_len or old_len + 1",
            element_type_change=elem_change,
        )

    @staticmethod
    def _dict_delitem(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="KeyError",
        )

    @staticmethod
    def _dict_update(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len <= new_len <= old_len + other_len",
            element_type_change=arg_types[0] if arg_types else None,
        )

    @staticmethod
    def _dict_pop(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        has_default = arg_types is not None and len(arg_types) >= 2
        return CollectionMutationResult(
            new_length_relation="old_len - 1 or old_len",
            may_raise=not has_default,
            exception_type="KeyError" if not has_default else None,
        )

    @staticmethod
    def _dict_popitem(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="KeyError",
        )

    @staticmethod
    def _dict_setdefault(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[1] if arg_types and len(arg_types) > 1 else None
        return CollectionMutationResult(
            new_length_relation="old_len or old_len + 1",
            element_type_change=elem_change,
        )

    @staticmethod
    def _dict_clear(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(new_length_relation="0")

    # ------------------------------------------------------------------
    # Set mutation handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_add(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        elem_change = arg_types[0] if arg_types else None
        return CollectionMutationResult(
            new_length_relation="old_len or old_len + 1",
            element_type_change=elem_change,
        )

    @staticmethod
    def _set_discard(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1 or old_len",
        )

    @staticmethod
    def _set_remove(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="KeyError",
        )

    @staticmethod
    def _set_pop(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len - 1",
            may_raise=True,
            exception_type="KeyError",
        )

    @staticmethod
    def _set_clear(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(new_length_relation="0")

    @staticmethod
    def _set_update(arg_types: Optional[List[str]] = None) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="old_len <= new_len <= old_len + other_len",
            element_type_change=arg_types[0] if arg_types else None,
        )

    @staticmethod
    def _set_intersection_update(
        arg_types: Optional[List[str]] = None,
    ) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="new_len <= old_len",
        )

    @staticmethod
    def _set_difference_update(
        arg_types: Optional[List[str]] = None,
    ) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="new_len <= old_len",
        )

    @staticmethod
    def _set_symmetric_difference_update(
        arg_types: Optional[List[str]] = None,
    ) -> CollectionMutationResult:
        return CollectionMutationResult(
            new_length_relation="abs(old_len - other_len) <= new_len <= old_len + other_len",
        )

    # ==================================================================
    # Public analysis API
    # ==================================================================

    # ------------------------------------------------------------------
    # analyze_setattr
    # ------------------------------------------------------------------

    def analyze_setattr(
        self,
        obj_type: str,
        attr: str,
        value_type: str,
        aliases: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[List[MutationEffect], List[RefinementInvalidation]]:
        """Analyse ``obj.attr = value`` for mutation effects and invalidations."""

        effects: List[MutationEffect] = []
        invalidations: List[RefinementInvalidation] = []

        # Check immutability first.
        imm = self.check_immutability(obj_type)
        if imm.is_immutable:
            # Attempting to set an attribute on an immutable type always
            # raises AttributeError at runtime, but for soundness we still
            # record the *attempt* so the caller can issue a diagnostic.
            invalidations.append(
                RefinementInvalidation(
                    variable=obj_type,
                    invalidated_attrs={attr},
                    reason=f"setattr on immutable type {obj_type}",
                )
            )
            return effects, invalidations

        # Try descriptor protocol: __set__ may intercede.
        resolution: Optional[AttributeResolution] = None
        try:
            resolution = self.descriptors.resolve_setattr(obj_type, attr, value_type)
        except (AttributeError, TypeError):
            pass

        descriptor_target = obj_type
        if resolution is not None and resolution.descriptor_info is not None:
            # The descriptor's __set__ might redirect the actual mutation.
            descriptor_target = resolution.descriptor_info.owner_type or obj_type

        effects.append(
            MutationEffect(
                target=descriptor_target,
                kind="setattr",
                attr_or_key=attr,
                value_type=value_type,
            )
        )

        # Invalidate refinements on the target and all its aliases.
        affected_vars = {obj_type}
        if aliases:
            for var, alias_set in aliases.items():
                if obj_type in alias_set or var == obj_type:
                    affected_vars |= alias_set
                    affected_vars.add(var)

        for var in affected_vars:
            invalidations.append(
                RefinementInvalidation(
                    variable=var,
                    invalidated_attrs={attr},
                    reason=f"setattr {attr} on {obj_type}",
                )
            )

        return effects, invalidations

    # ------------------------------------------------------------------
    # analyze_delattr
    # ------------------------------------------------------------------

    def analyze_delattr(
        self,
        obj_type: str,
        attr: str,
        aliases: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[List[MutationEffect], List[RefinementInvalidation]]:
        """Analyse ``del obj.attr``."""

        effects: List[MutationEffect] = []
        invalidations: List[RefinementInvalidation] = []

        imm = self.check_immutability(obj_type)
        if imm.is_immutable:
            invalidations.append(
                RefinementInvalidation(
                    variable=obj_type,
                    invalidated_attrs={attr},
                    reason=f"delattr on immutable type {obj_type}",
                )
            )
            return effects, invalidations

        # Descriptor protocol: __delete__ may intercede.
        try:
            resolution = self.descriptors.resolve_delattr(obj_type, attr)
            if resolution is not None and resolution.descriptor_info is not None:
                target = resolution.descriptor_info.owner_type or obj_type
            else:
                target = obj_type
        except (AttributeError, TypeError):
            target = obj_type

        effects.append(
            MutationEffect(target=target, kind="delattr", attr_or_key=attr)
        )

        affected_vars = self._expand_aliases(obj_type, aliases)
        for var in affected_vars:
            invalidations.append(
                RefinementInvalidation(
                    variable=var,
                    invalidated_attrs={attr},
                    reason=f"delattr {attr} on {obj_type}",
                )
            )

        return effects, invalidations

    # ------------------------------------------------------------------
    # analyze_setitem
    # ------------------------------------------------------------------

    def analyze_setitem(
        self,
        container_type: str,
        key_type: str,
        value_type: str,
        aliases: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[List[MutationEffect], List[RefinementInvalidation]]:
        """Analyse ``container[key] = value``."""

        effects: List[MutationEffect] = []
        invalidations: List[RefinementInvalidation] = []

        imm = self.check_immutability(container_type)
        if imm.is_immutable:
            invalidations.append(
                RefinementInvalidation(
                    variable=container_type,
                    invalidated_attrs=set(),
                    reason=f"setitem on immutable type {container_type}",
                )
            )
            return effects, invalidations

        effects.append(
            MutationEffect(
                target=container_type,
                kind="setitem",
                attr_or_key=key_type,
                value_type=value_type,
            )
        )

        affected_vars = self._expand_aliases(container_type, aliases)
        for var in affected_vars:
            # __setitem__ can change length, element types, and key-based
            # refinements, so we conservatively invalidate all attrs.
            invalidations.append(
                RefinementInvalidation(
                    variable=var,
                    invalidated_attrs={"__len__", "__getitem__", "__contains__"},
                    reason=f"setitem on {container_type}",
                )
            )

        return effects, invalidations

    # ------------------------------------------------------------------
    # analyze_delitem
    # ------------------------------------------------------------------

    def analyze_delitem(
        self,
        container_type: str,
        key_type: str,
        aliases: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[List[MutationEffect], List[RefinementInvalidation]]:
        """Analyse ``del container[key]``."""

        effects: List[MutationEffect] = []
        invalidations: List[RefinementInvalidation] = []

        imm = self.check_immutability(container_type)
        if imm.is_immutable:
            invalidations.append(
                RefinementInvalidation(
                    variable=container_type,
                    invalidated_attrs=set(),
                    reason=f"delitem on immutable type {container_type}",
                )
            )
            return effects, invalidations

        effects.append(
            MutationEffect(
                target=container_type,
                kind="delitem",
                attr_or_key=key_type,
            )
        )

        affected_vars = self._expand_aliases(container_type, aliases)
        for var in affected_vars:
            invalidations.append(
                RefinementInvalidation(
                    variable=var,
                    invalidated_attrs={"__len__", "__getitem__", "__contains__"},
                    reason=f"delitem on {container_type}",
                )
            )

        return effects, invalidations

    # ------------------------------------------------------------------
    # Collection-level analysis façades
    # ------------------------------------------------------------------

    def analyze_list_mutation(
        self,
        method: str,
        arg_types: Optional[List[str]] = None,
    ) -> CollectionMutationResult:
        handler = self._list_mutations.get(method)
        if handler is None:
            return CollectionMutationResult()
        return handler(arg_types)

    def analyze_dict_mutation(
        self,
        method: str,
        arg_types: Optional[List[str]] = None,
    ) -> CollectionMutationResult:
        handler = self._dict_mutations.get(method)
        if handler is None:
            return CollectionMutationResult()
        return handler(arg_types)

    def analyze_set_mutation(
        self,
        method: str,
        arg_types: Optional[List[str]] = None,
    ) -> CollectionMutationResult:
        handler = self._set_mutations.get(method)
        if handler is None:
            return CollectionMutationResult()
        return handler(arg_types)

    # ==================================================================
    # Function-level frame computation
    # ==================================================================

    def compute_frame(
        self,
        func: ast.FunctionDef,
        call_graph: Optional[Dict[str, ast.FunctionDef]] = None,
    ) -> ModificationFrame:
        """Compute a *ModificationFrame* that summarises what *func* may
        read or write.

        ``call_graph`` is an optional mapping from fully-qualified callee
        names to their AST nodes so that transitive effects can be resolved.
        """

        frame = ModificationFrame()

        # Collect parameter names (first argument of each arg).
        param_names: Set[str] = set()
        for arg in func.args.args:
            param_names.add(arg.arg)
        for arg in func.args.posonlyargs:
            param_names.add(arg.arg)
        for arg in func.args.kwonlyargs:
            param_names.add(arg.arg)
        if func.args.vararg:
            param_names.add(func.args.vararg.arg)
        if func.args.kwarg:
            param_names.add(func.args.kwarg.arg)

        # Extract global / nonlocal declarations.
        global_names = self._extract_global_writes(func)
        nonlocal_names = self._extract_nonlocal_writes(func)

        # Track aliases: variable → set of params it may point to.
        alias_map = self._track_aliases(func.body, param_names)

        for stmt in ast.walk(func):
            if isinstance(stmt, ast.stmt):
                self._analyze_stmt_effects(stmt, frame, param_names, alias_map, global_names)

        # Walk expressions for method-call mutations.
        for node in ast.walk(func):
            if isinstance(node, ast.expr):
                self._analyze_expr_effects(node, frame, param_names, alias_map)

        # Transitive effects via call graph.
        if call_graph:
            self._merge_transitive_effects(func, frame, param_names, call_graph)

        # Globals / nonlocals.
        for gname in global_names:
            frame.reads_globals.add(gname)
        for gname in nonlocal_names:
            frame.reads_globals.add(gname)

        return frame

    # ------------------------------------------------------------------
    # Statement-level effects
    # ------------------------------------------------------------------

    def _analyze_stmt_effects(
        self,
        stmt: ast.stmt,
        frame: ModificationFrame,
        param_names: Set[str],
        alias_map: Optional[Dict[str, Set[str]]] = None,
        global_names: Optional[Set[str]] = None,
    ) -> None:
        """Analyse a single statement for mutation effects."""

        if global_names is None:
            global_names = set()
        if alias_map is None:
            alias_map = {}

        # --- Attribute assignment: x.attr = ... --------------------------
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                self._handle_assignment_target(
                    target, frame, param_names, alias_map, global_names
                )
        elif isinstance(stmt, ast.AugAssign):
            self._handle_assignment_target(
                stmt.target, frame, param_names, alias_map, global_names
            )
        elif isinstance(stmt, ast.AnnAssign) and stmt.target is not None:
            self._handle_assignment_target(
                stmt.target, frame, param_names, alias_map, global_names
            )

        # --- Delete -------------------------------------------------------
        elif isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                if isinstance(target, ast.Attribute):
                    base = self._resolve_name(target.value)
                    if base and self._name_is_param(base, param_names, alias_map):
                        param = self._resolve_to_param(base, param_names, alias_map)
                        if param:
                            frame.mark_attr_modified(param, target.attr)
                elif isinstance(target, ast.Subscript):
                    base = self._resolve_name(target.value)
                    if base and self._name_is_param(base, param_names, alias_map):
                        param = self._resolve_to_param(base, param_names, alias_map)
                        if param:
                            frame.mark_param_modified(param)
                elif isinstance(target, ast.Name):
                    if target.id in global_names:
                        frame.mark_global_modified(target.id)

        # --- Raise --------------------------------------------------------
        elif isinstance(stmt, ast.Raise):
            exc_type = self._infer_exception_type(stmt)
            if exc_type:
                frame.may_raise.add(exc_type)

        # --- Global / nonlocal declarations are handled separately --------

    def _handle_assignment_target(
        self,
        target: ast.expr,
        frame: ModificationFrame,
        param_names: Set[str],
        alias_map: Dict[str, Set[str]],
        global_names: Set[str],
    ) -> None:
        """Process a single assignment target."""

        if isinstance(target, ast.Attribute):
            base = self._resolve_name(target.value)
            if base and self._name_is_param(base, param_names, alias_map):
                param = self._resolve_to_param(base, param_names, alias_map)
                if param:
                    frame.mark_attr_modified(param, target.attr)
        elif isinstance(target, ast.Subscript):
            base = self._resolve_name(target.value)
            if base and self._name_is_param(base, param_names, alias_map):
                param = self._resolve_to_param(base, param_names, alias_map)
                if param:
                    frame.mark_param_modified(param)
        elif isinstance(target, ast.Name):
            if target.id in global_names:
                frame.mark_global_modified(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._handle_assignment_target(elt, frame, param_names, alias_map, global_names)

    # ------------------------------------------------------------------
    # Expression-level effects (mutating method calls)
    # ------------------------------------------------------------------

    def _analyze_expr_effects(
        self,
        expr: ast.expr,
        frame: ModificationFrame,
        param_names: Set[str],
        alias_map: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        if alias_map is None:
            alias_map = {}

        if not isinstance(expr, ast.Call):
            # Track reads of parameter names.
            if isinstance(expr, ast.Name):
                if expr.id in param_names:
                    frame.reads_params.add(expr.id)
                elif self._name_is_param(expr.id, param_names, alias_map):
                    param = self._resolve_to_param(expr.id, param_names, alias_map)
                    if param:
                        frame.reads_params.add(param)
            return

        result = self._is_param_mutation(expr, param_names, alias_map)
        if result is not None:
            param, method = result
            frame.mark_param_modified(param)
            # If the method is a known collection mutation, record the
            # specific attribute.
            if method in _LIST_MUTATION_METHODS | _DICT_MUTATION_METHODS | _SET_MUTATION_METHODS:
                frame.mark_attr_modified(param, method)

    # ------------------------------------------------------------------
    # Alias tracking
    # ------------------------------------------------------------------

    def _track_aliases(
        self,
        stmts: List[ast.stmt],
        param_names: Set[str],
    ) -> Dict[str, Set[str]]:
        """Build a conservative mapping *local_name → {param_names}* for
        simple ``alias = param`` assignments.

        Only direct name-to-name assignments are tracked; more complex
        aliasing (e.g., ``alias = some_func(param)``) is not resolved.
        """

        aliases: Dict[str, Set[str]] = {}

        for node in ast.walk(ast.Module(body=list(stmts), type_ignores=[])):
            if isinstance(node, ast.Assign):
                # Only handle ``alias = name`` where name is a param or
                # known alias.
                if not isinstance(node.value, ast.Name):
                    continue
                source = node.value.id
                source_params: Set[str] = set()
                if source in param_names:
                    source_params.add(source)
                elif source in aliases:
                    source_params = aliases[source]
                if not source_params:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.setdefault(target.id, set()).update(source_params)

        return aliases

    # ------------------------------------------------------------------
    # Immutability checking
    # ------------------------------------------------------------------

    def check_immutability(
        self,
        type_name: str,
        hierarchy: Optional[ClassHierarchyAnalyzer] = None,
    ) -> ImmutabilityInfo:
        """Determine whether *type_name* is immutable."""

        hier = hierarchy or self.hierarchy

        # Fast path: known builtin immutable types.
        if type_name in self._immutable_types:
            return ImmutabilityInfo(
                is_immutable=True,
                reason=f"{type_name} is a builtin immutable type",
            )

        # Check for frozen dataclass / NamedTuple via hierarchy.
        try:
            class_info: Optional[ClassInfo] = hier.get_class(type_name)
        except (KeyError, AttributeError):
            class_info = None

        if class_info is not None:
            # Frozen dataclass.
            if getattr(class_info, "is_frozen_dataclass", False):
                return ImmutabilityInfo(
                    is_immutable=True,
                    reason=f"{type_name} is a frozen dataclass",
                )

            # NamedTuple.
            if getattr(class_info, "is_namedtuple", False):
                return ImmutabilityInfo(
                    is_immutable=True,
                    reason=f"{type_name} is a NamedTuple",
                )

            # __slots__ without any __set__ descriptors on writable slots.
            slots = getattr(class_info, "slots", None)
            if slots is not None:
                mutable: Set[str] = set()
                for slot in slots:
                    if not self._slot_is_readonly(type_name, slot):
                        mutable.add(slot)
                if not mutable:
                    return ImmutabilityInfo(
                        is_immutable=True,
                        reason=f"{type_name} uses __slots__ with no writable slots",
                    )
                return ImmutabilityInfo(
                    is_immutable=False,
                    reason=f"{type_name} has writable slots: {mutable}",
                    mutable_attrs=mutable,
                )

            # Check if all bases are immutable (structural immutability).
            bases = getattr(class_info, "bases", [])
            if bases and all(
                self.check_immutability(b, hier).is_immutable for b in bases
            ):
                # If the class defines no mutable instance attributes we
                # treat it as immutable.
                instance_attrs = getattr(class_info, "instance_attrs", set())
                if not instance_attrs:
                    return ImmutabilityInfo(
                        is_immutable=True,
                        reason=f"{type_name} extends only immutable bases with "
                               f"no new instance attributes",
                    )

        return ImmutabilityInfo(
            is_immutable=False,
            reason=f"{type_name} is not known to be immutable",
        )

    # ------------------------------------------------------------------
    # Surviving refinements
    # ------------------------------------------------------------------

    def get_surviving_refinements(
        self,
        mutations: List[MutationEffect],
        current_refinements: Dict[str, Set[str]],
        aliases: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Given a list of *mutations* and the currently held refinement
        predicates (mapping variable → set of predicate names), return
        the refinements that survive (are not invalidated).

        *aliases* maps each variable to the set of other variables that
        may point to the same heap object.
        """

        surviving: Dict[str, Set[str]] = {
            var: set(preds) for var, preds in current_refinements.items()
        }

        for mutation in mutations:
            target = mutation.target
            # Build the set of variables affected by this mutation.
            affected: Set[str] = {target}
            for var, alias_set in aliases.items():
                if target in alias_set or var == target:
                    affected.add(var)
                    affected |= alias_set

            for var in affected:
                if var not in surviving:
                    continue

                if mutation.kind in ("clear",):
                    # Complete invalidation.
                    surviving[var] = set()
                elif mutation.kind in ("setattr", "delattr"):
                    attr = mutation.attr_or_key
                    if attr is not None:
                        surviving[var] = {
                            p for p in surviving[var]
                            if not self._predicate_depends_on_attr(p, attr)
                        }
                    else:
                        surviving[var] = set()
                elif mutation.kind in (
                    "setitem", "delitem", "append", "extend", "insert",
                    "pop", "remove", "update", "add", "discard",
                ):
                    # Length-dependent and element-dependent predicates are
                    # invalidated.
                    surviving[var] = {
                        p for p in surviving[var]
                        if not self._predicate_depends_on_content(p)
                    }
                elif mutation.kind in ("sort", "reverse"):
                    # Length is preserved; only order-dependent predicates
                    # are invalidated.
                    surviving[var] = {
                        p for p in surviving[var]
                        if not self._predicate_depends_on_order(p)
                    }
                else:
                    # Unknown mutation → conservatively invalidate all.
                    surviving[var] = set()

        return surviving

    # ------------------------------------------------------------------
    # Weakest-precondition calculus
    # ------------------------------------------------------------------

    def compute_weakest_precondition(
        self,
        mutation: MutationEffect,
        postcondition: str,
    ) -> Optional[str]:
        """Compute the weakest precondition of *postcondition* with respect
        to *mutation*.

        Returns ``None`` when the wp cannot be computed (i.e., the
        postcondition is invalidated entirely).
        """

        target = mutation.target
        kind = mutation.kind
        attr = mutation.attr_or_key
        vtype = mutation.value_type

        # ``x.attr = v`` — postconditions that mention ``x.attr`` need
        # substitution.
        if kind == "setattr" and attr is not None:
            accessor = f"{target}.{attr}"
            if accessor in postcondition:
                if vtype is not None:
                    return postcondition.replace(accessor, f"({vtype})")
                return None  # can't compute without value type

            # Postconditions that don't mention the modified attribute
            # survive unchanged.
            return postcondition

        if kind == "delattr" and attr is not None:
            accessor = f"{target}.{attr}"
            if accessor in postcondition:
                return None
            return postcondition

        if kind == "setitem":
            # ``x[k] = v`` — postconditions mentioning x[...] or len(x)
            # may be affected.
            if f"len({target})" in postcondition:
                return None
            if f"{target}[" in postcondition:
                return None
            return postcondition

        if kind == "delitem":
            if f"len({target})" in postcondition or f"{target}[" in postcondition:
                return None
            return postcondition

        if kind in ("append", "extend", "insert", "pop", "remove"):
            if f"len({target})" in postcondition:
                return None
            if f"{target}[" in postcondition:
                return None
            return postcondition

        if kind == "clear":
            if target in postcondition:
                return None
            return postcondition

        if kind in ("sort", "reverse"):
            # Length is preserved.
            if f"{target}[" in postcondition:
                return None
            return postcondition

        if kind in ("update", "add", "discard"):
            if f"len({target})" in postcondition or f"{target}" in postcondition:
                return None
            return postcondition

        return None

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _is_param_mutation(
        self,
        node: ast.expr,
        param_names: Set[str],
        alias_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Check whether *node* (a Call) is a mutating method call on a
        parameter (or alias thereof).

        Returns ``(param_name, method_name)`` or ``None``.
        """

        if alias_map is None:
            alias_map = {}

        if not isinstance(node, ast.Call):
            return None

        func = node.func
        if not isinstance(func, ast.Attribute):
            return None

        method = func.attr
        base = self._resolve_name(func.value)
        if base is None:
            return None

        # Direct parameter.
        if base in param_names:
            if method in (
                _LIST_MUTATION_METHODS | _DICT_MUTATION_METHODS | _SET_MUTATION_METHODS
            ):
                return (base, method)

        # Alias of a parameter.
        if base in alias_map:
            for param in alias_map[base]:
                if param in param_names:
                    if method in (
                        _LIST_MUTATION_METHODS
                        | _DICT_MUTATION_METHODS
                        | _SET_MUTATION_METHODS
                    ):
                        return (param, method)

        return None

    # ------------------------------------------------------------------
    # Global / nonlocal extraction
    # ------------------------------------------------------------------

    def _extract_global_writes(self, func: ast.FunctionDef) -> Set[str]:
        """Return names declared ``global`` inside *func*."""

        names: Set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Global):
                names.update(node.names)
        return names

    def _extract_nonlocal_writes(self, func: ast.FunctionDef) -> Set[str]:
        """Return names declared ``nonlocal`` inside *func*."""

        names: Set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Nonlocal):
                names.update(node.names)
        return names

    # ------------------------------------------------------------------
    # Alias / name resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_name(node: ast.expr) -> Optional[str]:
        """Extract a simple name from an expression node."""
        if isinstance(node, ast.Name):
            return node.id
        return None

    @staticmethod
    def _name_is_param(
        name: str,
        param_names: Set[str],
        alias_map: Dict[str, Set[str]],
    ) -> bool:
        if name in param_names:
            return True
        if name in alias_map:
            return bool(alias_map[name] & param_names)
        return False

    @staticmethod
    def _resolve_to_param(
        name: str,
        param_names: Set[str],
        alias_map: Dict[str, Set[str]],
    ) -> Optional[str]:
        """Resolve *name* to a parameter name via *alias_map*."""
        if name in param_names:
            return name
        if name in alias_map:
            candidates = alias_map[name] & param_names
            if candidates:
                return next(iter(candidates))
        return None

    @staticmethod
    def _expand_aliases(
        name: str,
        aliases: Optional[Dict[str, Set[str]]],
    ) -> Set[str]:
        """Return *name* plus all variables that may alias it."""
        affected: Set[str] = {name}
        if aliases:
            for var, alias_set in aliases.items():
                if name in alias_set or var == name:
                    affected.add(var)
                    affected |= alias_set
        return affected

    # ------------------------------------------------------------------
    # Exception type inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_exception_type(stmt: ast.Raise) -> Optional[str]:
        """Best-effort extraction of the exception type from a Raise node."""
        if stmt.exc is None:
            return "BaseException"
        if isinstance(stmt.exc, ast.Call):
            if isinstance(stmt.exc.func, ast.Name):
                return stmt.exc.func.id
            if isinstance(stmt.exc.func, ast.Attribute):
                return stmt.exc.func.attr
        if isinstance(stmt.exc, ast.Name):
            return stmt.exc.id
        return "Exception"

    # ------------------------------------------------------------------
    # Slot read-only check
    # ------------------------------------------------------------------

    def _slot_is_readonly(self, type_name: str, slot: str) -> bool:
        """Return True when *slot* on *type_name* is read-only (i.e.,
        guarded by a descriptor with no ``__set__``).
        """
        try:
            resolution = self.descriptors.resolve_getattr(type_name, slot)
            if resolution is not None and resolution.descriptor_info is not None:
                return not getattr(resolution.descriptor_info, "has_set", True)
        except (AttributeError, TypeError, KeyError):
            pass
        return False

    # ------------------------------------------------------------------
    # Transitive effect merging
    # ------------------------------------------------------------------

    def _merge_transitive_effects(
        self,
        func: ast.FunctionDef,
        frame: ModificationFrame,
        param_names: Set[str],
        call_graph: Dict[str, ast.FunctionDef],
    ) -> None:
        """Walk *func* for call expressions and, when the callee is present
        in *call_graph*, recursively compute its frame and merge.
        """

        visited: Set[str] = set()

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            callee_name = self._extract_callee_name(node)
            if callee_name is None or callee_name in visited:
                continue
            if callee_name not in call_graph:
                continue
            visited.add(callee_name)

            callee_func = call_graph[callee_name]
            callee_frame = self.compute_frame(callee_func, call_graph=None)
            # Map callee param modifications back to caller params.
            arg_mapping = self._build_arg_mapping(node, callee_func, param_names)
            mapped_frame = self._remap_frame(callee_frame, arg_mapping)
            frame.merge(mapped_frame)

    @staticmethod
    def _extract_callee_name(call: ast.Call) -> Optional[str]:
        """Get the qualified name of a Call target."""
        func = call.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = MutationSemantics._resolve_name(func.value)
            if base:
                return f"{base}.{func.attr}"
        return None

    @staticmethod
    def _build_arg_mapping(
        call: ast.Call,
        callee: ast.FunctionDef,
        caller_params: Set[str],
    ) -> Dict[str, str]:
        """Map callee parameter names to caller argument names (when the
        argument is a simple Name referencing a caller parameter).
        """

        mapping: Dict[str, str] = {}
        callee_params = [a.arg for a in callee.args.args]

        for i, arg in enumerate(call.args):
            if i < len(callee_params) and isinstance(arg, ast.Name):
                if arg.id in caller_params:
                    mapping[callee_params[i]] = arg.id

        for kw in call.keywords:
            if kw.arg and isinstance(kw.value, ast.Name):
                if kw.value.id in caller_params:
                    mapping[kw.arg] = kw.value.id

        return mapping

    @staticmethod
    def _remap_frame(
        callee_frame: ModificationFrame,
        arg_mapping: Dict[str, str],
    ) -> ModificationFrame:
        """Produce a new frame where callee param references are replaced
        with caller param references according to *arg_mapping*.
        """

        remapped = ModificationFrame()

        for p in callee_frame.modified_params:
            if p in arg_mapping:
                remapped.modified_params.add(arg_mapping[p])
        for p, attrs in callee_frame.modified_attrs.items():
            if p in arg_mapping:
                remapped.modified_attrs.setdefault(arg_mapping[p], set()).update(attrs)

        remapped.modified_globals = set(callee_frame.modified_globals)
        remapped.may_modify_globals = callee_frame.may_modify_globals
        remapped.may_raise = set(callee_frame.may_raise)
        remapped.pure = callee_frame.pure
        remapped.reads_globals = set(callee_frame.reads_globals)

        for p in callee_frame.reads_params:
            if p in arg_mapping:
                remapped.reads_params.add(arg_mapping[p])

        return remapped

    # ------------------------------------------------------------------
    # Predicate classification helpers (for surviving-refinement logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _predicate_depends_on_attr(predicate: str, attr: str) -> bool:
        """Conservative check: does *predicate* textually reference *attr*?"""
        return f".{attr}" in predicate or attr in predicate

    @staticmethod
    def _predicate_depends_on_content(predicate: str) -> bool:
        """Does *predicate* depend on collection content/length?"""
        content_keywords = {"len", "contains", "in", "[", "elem", "item", "size"}
        pred_lower = predicate.lower()
        return any(kw in pred_lower for kw in content_keywords)

    @staticmethod
    def _predicate_depends_on_order(predicate: str) -> bool:
        """Does *predicate* depend on element ordering?"""
        order_keywords = {"[0]", "[1]", "[-1]", "first", "last", "sorted", "index"}
        pred_lower = predicate.lower()
        return any(kw in pred_lower for kw in order_keywords)
