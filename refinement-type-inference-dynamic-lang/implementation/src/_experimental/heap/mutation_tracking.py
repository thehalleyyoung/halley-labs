"""
Mutation tracking for heap-aware refinement invalidation.

Tracks writes, escapes, and immutability to determine which active
refinements must be killed when the heap is modified.
"""
from __future__ import annotations

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

from src.heap.heap_model import HeapAddress, AbstractValue, AbstractHeap, HeapObject
from src.heap.alias_analysis import AliasSet, FieldPath


# ---------------------------------------------------------------------------
# 1. MutationKind
# ---------------------------------------------------------------------------

class MutationKind(Enum):
    """Categories of heap-mutating operations."""
    ATTR_WRITE = auto()          # x.attr = val
    SUBSCRIPT_WRITE = auto()     # x[k] = val
    IN_PLACE_MUTATION = auto()   # x.append(v), x.update(...)
    DELETE_ATTR = auto()         # del x.attr
    DELETE_SUBSCRIPT = auto()    # del x[k]
    BULK_MUTATION = auto()       # x.clear(), x.sort()
    UNKNOWN_ESCAPE = auto()      # arg passed to unknown function


# ---------------------------------------------------------------------------
# 2. MutationEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationEvent:
    """A single mutation observed during analysis."""
    kind: MutationKind
    target_addr: HeapAddress
    field_name: Optional[str] = None
    new_value: Optional[AbstractValue] = None
    program_point: int = -1


# ---------------------------------------------------------------------------
# 3. RefinementRef
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementRef:
    """Reference to an active refinement predicate bound to a variable/path."""
    variable: str
    field_path: Tuple[str, ...] = ()
    predicate_id: str = ""

    def reaches_field(self, field_name: str) -> bool:
        """Return True if this refinement depends on *field_name*.

        A refinement depends on a field when:
          - Its field_path starts with that field (goes *through* it), or
          - Its field_path is exactly (field_name,).
        """
        if not self.field_path:
            return False
        return self.field_path[0] == field_name

    def is_length_refinement(self) -> bool:
        """Heuristic: predicate id contains 'len' or 'length'."""
        pid = self.predicate_id.lower()
        return "len" in pid or "length" in pid

    def is_order_dependent(self) -> bool:
        """Heuristic: predicate references index-based access or ordering."""
        pid = self.predicate_id.lower()
        return any(tok in pid for tok in ("index", "sorted", "order", "first", "last"))

    def is_element_type_refinement(self) -> bool:
        """Heuristic: predicate is about element types (e.g. List[int])."""
        pid = self.predicate_id.lower()
        return any(tok in pid for tok in ("elem", "element", "item_type", "contains_type"))


# ---------------------------------------------------------------------------
# 4. MutationTracker
# ---------------------------------------------------------------------------

class MutationTracker:
    """Main mutation tracking system.

    Maintains the set of active refinements and, on each mutation event,
    returns which refinements are invalidated (killed).
    """

    def __init__(self, alias_set: AliasSet) -> None:
        self.alias_set: AliasSet = alias_set
        # variable name -> set of refinements that mention it
        self.active_refinements: Dict[str, Set[RefinementRef]] = {}
        self.immutable_addrs: Set[HeapAddress] = set()
        self.escape_set: Set[HeapAddress] = set()
        self._history: List[MutationEvent] = []

    # -- refinement bookkeeping ---------------------------------------------

    def add_refinement(self, ref: RefinementRef) -> None:
        """Register a new active refinement."""
        self.active_refinements.setdefault(ref.variable, set()).add(ref)

    def remove_refinement(self, ref: RefinementRef) -> None:
        """Remove a specific refinement."""
        refs = self.active_refinements.get(ref.variable)
        if refs is not None:
            refs.discard(ref)
            if not refs:
                del self.active_refinements[ref.variable]

    def get_active_refinements(self, var: str) -> Set[RefinementRef]:
        """Return all active refinements for *var*."""
        return set(self.active_refinements.get(var, ()))

    # -- immutability / escape bookkeeping ----------------------------------

    def mark_immutable(self, addr: HeapAddress) -> None:
        self.immutable_addrs.add(addr)

    def is_immutable(self, addr: HeapAddress) -> bool:
        return addr in self.immutable_addrs

    def mark_escaped(self, addr: HeapAddress) -> None:
        self.escape_set.add(addr)

    def has_escaped(self, addr: HeapAddress) -> bool:
        return addr in self.escape_set

    # -- helpers ------------------------------------------------------------

    def _variables_for_addr(self, addr: HeapAddress) -> Set[str]:
        """Return all variable names that may alias *addr*."""
        try:
            return self.alias_set.variables_for(addr)
        except AttributeError:
            # Fallback: iterate the alias set's mapping if the method is
            # spelled differently in the concrete AliasSet implementation.
            mapping = getattr(self.alias_set, "addr_to_vars", {})
            return set(mapping.get(addr, ()))

    def _kill_matching(
        self,
        variables: Set[str],
        predicate,
    ) -> Set[RefinementRef]:
        """Remove and return all refinements in *variables* matching *predicate*."""
        killed: Set[RefinementRef] = set()
        for var in variables:
            refs = self.active_refinements.get(var)
            if refs is None:
                continue
            to_kill = {r for r in refs if predicate(r)}
            killed |= to_kill
            refs -= to_kill
            if not refs:
                self.active_refinements.pop(var, None)
        return killed

    def _record(self, event: MutationEvent) -> None:
        self._history.append(event)

    # -- mutation handlers --------------------------------------------------

    def on_setattr(
        self,
        addr: HeapAddress,
        name: str,
        heap: AbstractHeap,
    ) -> Set[RefinementRef]:
        """Handle ``x.attr = val``.  Returns invalidated refinements."""
        if self.is_immutable(addr):
            return set()

        event = MutationEvent(
            kind=MutationKind.ATTR_WRITE,
            target_addr=addr,
            field_name=name,
        )
        self._record(event)

        aliased_vars = self._variables_for_addr(addr)
        # Kill refinements whose field_path starts with the mutated field,
        # or that directly reference *name*.
        killed = self._kill_matching(
            aliased_vars,
            lambda r: r.reaches_field(name),
        )
        # Also kill any refinement on the variable itself whose predicate
        # mentions the field name (e.g. hasattr(x, 'foo')).
        killed |= self._kill_matching(
            aliased_vars,
            lambda r: (
                not r.field_path
                and name in r.predicate_id
            ),
        )
        return killed

    def on_getattr(
        self,
        addr: HeapAddress,
        name: str,
        heap: AbstractHeap,
    ) -> None:
        """Handle ``x.attr`` read — no invalidation, but we may want to
        record access for later escape analysis or liveness."""
        pass

    def on_call(
        self,
        func_addr: HeapAddress,
        args: List[HeapAddress],
        heap: AbstractHeap,
        escaping_params: Optional[Set[int]] = None,
    ) -> Set[RefinementRef]:
        """Handle a function call.

        If *escaping_params* is ``None`` (no summary available), conservatively
        assume every argument may be mutated.  Otherwise only invalidate
        refinements on the escaping arguments.
        """
        if escaping_params is None:
            escaping_indices: Set[int] = set(range(len(args)))
        else:
            escaping_indices = escaping_params

        killed: Set[RefinementRef] = set()
        for idx in escaping_indices:
            if idx >= len(args):
                continue
            arg_addr = args[idx]
            if self.is_immutable(arg_addr):
                continue
            self.mark_escaped(arg_addr)

            event = MutationEvent(
                kind=MutationKind.UNKNOWN_ESCAPE,
                target_addr=arg_addr,
            )
            self._record(event)

            aliased_vars = self._variables_for_addr(arg_addr)
            # Conservative: kill ALL refinements on aliased variables.
            killed |= self._kill_matching(aliased_vars, lambda _r: True)
        return killed

    def on_store_subscript(
        self,
        container: HeapAddress,
        key: Optional[AbstractValue],
        val: Optional[AbstractValue],
    ) -> Set[RefinementRef]:
        """Handle ``x[k] = val``.

        Heuristic preservation rules:
          - Length refinements are kept when overwriting an existing key in a
            list (index within known bounds).
          - Length refinements are killed when the container is a dict and the
            key may be new.
        """
        if self.is_immutable(container):
            return set()

        event = MutationEvent(
            kind=MutationKind.SUBSCRIPT_WRITE,
            target_addr=container,
            new_value=val,
        )
        self._record(event)

        aliased_vars = self._variables_for_addr(container)

        # Determine whether this write could change the container's length.
        key_is_new = self._key_may_be_new(container, key)

        def should_kill(r: RefinementRef) -> bool:
            if r.is_length_refinement() and not key_is_new:
                # Overwriting an existing slot doesn't change length.
                return False
            if r.is_element_type_refinement():
                # Element-type refinements killed only if new value
                # is potentially incompatible — conservatively kill.
                return True
            # Order-dependent refinements are killed unconditionally.
            if r.is_order_dependent():
                return True
            # Refinements that don't depend on subscript contents survive.
            if r.field_path:
                return False
            # Generic refinements on the container — kill conservatively.
            return True

        return self._kill_matching(aliased_vars, should_kill)

    def on_delete_attr(
        self,
        addr: HeapAddress,
        name: str,
    ) -> Set[RefinementRef]:
        """Handle ``del x.attr``."""
        if self.is_immutable(addr):
            return set()

        event = MutationEvent(
            kind=MutationKind.DELETE_ATTR,
            target_addr=addr,
            field_name=name,
        )
        self._record(event)

        aliased_vars = self._variables_for_addr(addr)
        killed = self._kill_matching(
            aliased_vars,
            lambda r: r.reaches_field(name),
        )
        # Also kill hasattr-style refinements.
        killed |= self._kill_matching(
            aliased_vars,
            lambda r: not r.field_path and name in r.predicate_id,
        )
        return killed

    def on_delete_subscript(
        self,
        container: HeapAddress,
        key: Optional[AbstractValue],
    ) -> Set[RefinementRef]:
        """Handle ``del x[k]``."""
        if self.is_immutable(container):
            return set()

        event = MutationEvent(
            kind=MutationKind.DELETE_SUBSCRIPT,
            target_addr=container,
        )
        self._record(event)

        aliased_vars = self._variables_for_addr(container)
        # Deletion always changes length and may re-index elements.
        return self._kill_matching(aliased_vars, lambda _r: True)

    def on_bulk_mutation(
        self,
        addr: HeapAddress,
        method_name: str,
    ) -> Set[RefinementRef]:
        """Handle in-place bulk mutations like clear/sort/append/extend."""
        if self.is_immutable(addr):
            return set()

        event = MutationEvent(
            kind=MutationKind.BULK_MUTATION,
            target_addr=addr,
            field_name=method_name,
        )
        self._record(event)

        aliased_vars = self._variables_for_addr(addr)

        lower = method_name.lower()

        if lower == "clear":
            # Everything is invalid after a clear.
            return self._kill_matching(aliased_vars, lambda _r: True)

        if lower == "sort" or lower == "reverse":
            # Length is preserved, element types preserved; order changes.
            return self._kill_matching(
                aliased_vars,
                lambda r: (
                    r.is_order_dependent()
                    or (not r.is_length_refinement()
                        and not r.is_element_type_refinement())
                ),
            )

        if lower in ("append", "extend", "insert"):
            # Length changes; element types may change.
            return self._kill_matching(
                aliased_vars,
                lambda r: r.is_length_refinement() or r.is_order_dependent(),
            )

        if lower in ("pop", "remove"):
            # Length changes, order may change.
            return self._kill_matching(
                aliased_vars,
                lambda r: r.is_length_refinement() or r.is_order_dependent(),
            )

        if lower == "update":
            # Dict.update — may add new keys, change values.
            return self._kill_matching(aliased_vars, lambda _r: True)

        if lower in ("add", "discard"):
            # Set operations — length may change.
            return self._kill_matching(
                aliased_vars,
                lambda r: r.is_length_refinement(),
            )

        # Unknown method — conservative.
        return self._kill_matching(aliased_vars, lambda _r: True)

    # -- private helpers ----------------------------------------------------

    def _key_may_be_new(
        self,
        container: HeapAddress,
        key: Optional[AbstractValue],
    ) -> bool:
        """Conservative estimate: could *key* introduce a new slot?

        Without concrete key knowledge we assume yes.
        """
        if key is None:
            return True
        # If key has a concrete integer value within a known list length we
        # could prove it is existing, but we lack runtime info here —
        # conservatively return True.
        return True


# ---------------------------------------------------------------------------
# 5. FrameCondition
# ---------------------------------------------------------------------------

@dataclass
class FrameCondition:
    """Summarises the side-effects of a function.

    A *frame condition* describes what a callee may or must not modify,
    enabling the caller to selectively invalidate refinements rather than
    killing everything.
    """
    may_modify: Set[FieldPath] = field(default_factory=set)
    must_not_modify: Set[FieldPath] = field(default_factory=set)
    may_allocate: bool = False
    may_raise: Set[str] = field(default_factory=set)
    modifies_global: Set[str] = field(default_factory=set)
    pure: bool = False

    # -- queries ------------------------------------------------------------

    def invalidated_by(self, event: MutationEvent) -> bool:
        """Return True if *event* conflicts with what this frame promises."""
        if self.pure:
            # A pure function cannot have generated *event* — so if we are
            # checking whether the frame itself is violated, any mutation
            # event means the summary is wrong / event is external.
            return True

        if event.field_name is not None:
            target_path = FieldPath((event.field_name,))
            if target_path in self.must_not_modify:
                return True

            if self.may_modify and target_path not in self.may_modify:
                # The function does not list this path as modifiable —
                # the event does not originate from this function.
                return False

        # If the event is a bulk mutation or unknown escape, and we have no
        # explicit must_not_modify, conservatively say yes.
        if event.kind in (
            MutationKind.BULK_MUTATION,
            MutationKind.UNKNOWN_ESCAPE,
        ):
            return True

        return bool(self.may_modify)

    # -- lattice operations -------------------------------------------------

    def join(self, other: FrameCondition) -> FrameCondition:
        """Conservative union (over-approximation of both)."""
        return FrameCondition(
            may_modify=self.may_modify | other.may_modify,
            must_not_modify=self.must_not_modify & other.must_not_modify,
            may_allocate=self.may_allocate or other.may_allocate,
            may_raise=self.may_raise | other.may_raise,
            modifies_global=self.modifies_global | other.modifies_global,
            pure=self.pure and other.pure,
        )

    def meet(self, other: FrameCondition) -> FrameCondition:
        """Under-approximation (intersection)."""
        return FrameCondition(
            may_modify=self.may_modify & other.may_modify,
            must_not_modify=self.must_not_modify | other.must_not_modify,
            may_allocate=self.may_allocate and other.may_allocate,
            may_raise=self.may_raise & other.may_raise,
            modifies_global=self.modifies_global & other.modifies_global,
            pure=self.pure or other.pure,
        )

    @staticmethod
    def from_function_summary(
        modified_paths: Set[FieldPath],
        escape_info: Optional[Set[HeapAddress]] = None,
    ) -> FrameCondition:
        """Build a FrameCondition from an inter-procedural summary.

        *modified_paths* are the field paths the function body writes to.
        *escape_info* is the set of addresses that escape through the
        function (used to decide ``may_allocate``).
        """
        pure = len(modified_paths) == 0 and (
            escape_info is None or len(escape_info) == 0
        )
        return FrameCondition(
            may_modify=set(modified_paths),
            must_not_modify=set(),
            may_allocate=escape_info is not None and len(escape_info) > 0,
            may_raise=set(),
            modifies_global=set(),
            pure=pure,
        )


# ---------------------------------------------------------------------------
# 6. ImmutabilityAnalyzer
# ---------------------------------------------------------------------------

# Built-in immutable type names (fully-qualified or short).
_BUILTIN_IMMUTABLE: FrozenSet[str] = frozenset({
    "int", "float", "complex", "bool", "str", "bytes",
    "tuple", "frozenset", "NoneType", "range", "type",
    "builtins.int", "builtins.float", "builtins.complex",
    "builtins.bool", "builtins.str", "builtins.bytes",
    "builtins.tuple", "builtins.frozenset", "builtins.NoneType",
    "builtins.range", "builtins.type",
})


class ImmutabilityAnalyzer:
    """Determine whether objects or types are immutable."""

    @staticmethod
    def is_immutable_type(
        class_addr: HeapAddress,
        registry: Any,
    ) -> bool:
        """Check if the class at *class_addr* denotes an immutable type.

        *registry* is expected to expose:
          - ``get_type_name(addr) -> str``
          - ``get_class_meta(addr) -> dict`` with optional keys
            ``frozen``, ``has_slots``, ``has_setters``.
        """
        type_name: str = ""
        try:
            type_name = registry.get_type_name(class_addr)
        except (AttributeError, KeyError):
            pass

        if type_name in _BUILTIN_IMMUTABLE:
            return True

        # User-defined frozen dataclass / NamedTuple.
        try:
            meta = registry.get_class_meta(class_addr)
        except (AttributeError, KeyError):
            return False

        if not isinstance(meta, dict):
            return False

        if meta.get("frozen", False):
            return True

        # Class with __slots__ and no property setters.
        if meta.get("has_slots", False) and not meta.get("has_setters", False):
            return True

        return False

    @staticmethod
    def is_effectively_immutable(
        addr: HeapAddress,
        heap: AbstractHeap,
        tracker: MutationTracker,
    ) -> bool:
        """Return True if *addr* is never mutated within the analysis scope.

        This is determined by scanning the tracker's mutation history for any
        event targeting *addr* or any of its aliases.
        """
        if tracker.is_immutable(addr):
            return True

        # Check the recorded history.
        aliased_addrs: Set[HeapAddress] = set()
        aliased_addrs.add(addr)
        try:
            aliased_addrs |= tracker.alias_set.aliases_of(addr)
        except AttributeError:
            pass

        for event in tracker._history:
            if event.target_addr in aliased_addrs:
                return False

        return True

    @staticmethod
    def immutable_fields(
        class_addr: HeapAddress,
        registry: Any,
    ) -> Set[str]:
        """Return the set of field names that are never written after ``__init__``.

        Relies on ``registry.get_field_writes(class_addr)`` returning a dict
        mapping field names to lists of write locations.  A field is
        "immutable" if all its writes occur inside ``__init__``.
        """
        result: Set[str] = set()
        try:
            field_writes: Dict[str, List[str]] = registry.get_field_writes(class_addr)
        except (AttributeError, KeyError):
            return result

        for fname, locations in field_writes.items():
            if all(loc == "__init__" for loc in locations):
                result.add(fname)

        return result


# ---------------------------------------------------------------------------
# 7. EscapeAnalyzer
# ---------------------------------------------------------------------------

class EscapeAnalyzer:
    """Track which heap addresses escape analysis scope.

    An address *escapes* if it is reachable from outside the current
    function after the function returns (or during execution via unknown
    callees).
    """

    @staticmethod
    def analyze_escaping(
        func_body: Any,
        alias_set: AliasSet,
    ) -> Set[HeapAddress]:
        """Compute the set of addresses that escape *func_body*.

        *func_body* is expected to expose:
          - ``return_values: List[HeapAddress]``
          - ``global_stores: List[Tuple[str, HeapAddress]]``
          - ``call_sites: List[Tuple[HeapAddress, List[HeapAddress]]]``
            (callee_addr, argument addrs)
          - ``attr_stores: List[Tuple[HeapAddress, str, HeapAddress]]``
            (target, attr_name, value_addr)
        """
        escaping: Set[HeapAddress] = set()

        # 1. Returns
        return_values: List[HeapAddress] = getattr(func_body, "return_values", [])
        for addr in return_values:
            escaping.add(addr)

        # 2. Global stores
        global_stores = getattr(func_body, "global_stores", [])
        for _gname, addr in global_stores:
            escaping.add(addr)

        # 3. Calls to unknown functions
        call_sites = getattr(func_body, "call_sites", [])
        for callee, call_args in call_sites:
            # If the callee is unknown, every argument escapes.
            if not _is_known_callee(callee, alias_set):
                for arg_addr in call_args:
                    escaping.add(arg_addr)

        # 4. Stored as attribute of an already-escaping object (first pass).
        attr_stores = getattr(func_body, "attr_stores", [])
        for target, _aname, val_addr in attr_stores:
            if target in escaping:
                escaping.add(val_addr)

        # 5. Resolve aliases.
        expanded: Set[HeapAddress] = set()
        for addr in escaping:
            expanded.add(addr)
            try:
                expanded |= alias_set.aliases_of(addr)
            except AttributeError:
                pass
        escaping = expanded

        # 6. Transitive closure via attr_stores.
        changed = True
        while changed:
            changed = False
            for target, _aname, val_addr in attr_stores:
                if target in escaping and val_addr not in escaping:
                    escaping.add(val_addr)
                    changed = True

        return escaping

    @staticmethod
    def escapes_through_return(
        addr: HeapAddress,
        alias_set: AliasSet,
    ) -> bool:
        """Check whether *addr* (or an alias) is returned from the function."""
        try:
            return_set: Set[HeapAddress] = alias_set.return_set()
        except AttributeError:
            return False

        if addr in return_set:
            return True

        try:
            aliases = alias_set.aliases_of(addr)
        except AttributeError:
            aliases = set()

        return bool(aliases & return_set)

    @staticmethod
    def escapes_through_global(
        addr: HeapAddress,
        alias_set: AliasSet,
    ) -> bool:
        """Check whether *addr* is stored in a global variable."""
        try:
            global_addrs: Set[HeapAddress] = alias_set.global_store_targets()
        except AttributeError:
            return False

        if addr in global_addrs:
            return True

        try:
            aliases = alias_set.aliases_of(addr)
        except AttributeError:
            aliases = set()

        return bool(aliases & global_addrs)

    @staticmethod
    def escapes_through_call(
        addr: HeapAddress,
        call_args: List[HeapAddress],
        alias_set: AliasSet,
    ) -> bool:
        """Return True if *addr* appears among *call_args* (directly or via alias)."""
        if addr in call_args:
            return True

        try:
            aliases = alias_set.aliases_of(addr)
        except AttributeError:
            aliases = set()

        return bool(aliases & set(call_args))

    @staticmethod
    def transitive_escape(
        initial_escaping: Set[HeapAddress],
        heap: AbstractHeap,
    ) -> Set[HeapAddress]:
        """Compute the transitive closure of escaping addresses.

        If *x* escapes and *x.foo* points to *y*, then *y* also escapes.
        """
        result: Set[HeapAddress] = set(initial_escaping)
        worklist: List[HeapAddress] = list(initial_escaping)

        while worklist:
            addr = worklist.pop()
            obj: Optional[HeapObject] = _get_heap_object(heap, addr)
            if obj is None:
                continue

            children = _children_of(obj)
            for child_addr in children:
                if child_addr not in result:
                    result.add(child_addr)
                    worklist.append(child_addr)

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_known_callee(callee: HeapAddress, alias_set: AliasSet) -> bool:
    """Return True if the callee has a known summary / is analysed."""
    try:
        return alias_set.has_summary(callee)
    except AttributeError:
        return False


def _get_heap_object(
    heap: AbstractHeap,
    addr: HeapAddress,
) -> Optional[HeapObject]:
    """Safely retrieve a HeapObject from the abstract heap."""
    try:
        return heap.get(addr)
    except (KeyError, AttributeError):
        return None


def _children_of(obj: HeapObject) -> Set[HeapAddress]:
    """Return all HeapAddress values reachable from *obj*'s fields."""
    children: Set[HeapAddress] = set()

    # Try the canonical ``fields`` mapping.
    fields = getattr(obj, "fields", None)
    if isinstance(fields, dict):
        for val in fields.values():
            if isinstance(val, HeapAddress):
                children.add(val)
            # If the value is an AbstractValue wrapping an address, try to
            # extract it.
            wrapped = getattr(val, "address", None)
            if isinstance(wrapped, HeapAddress):
                children.add(wrapped)

    # Try an explicit ``children`` method.
    child_method = getattr(obj, "children", None)
    if callable(child_method):
        for c in child_method():
            if isinstance(c, HeapAddress):
                children.add(c)

    return children
