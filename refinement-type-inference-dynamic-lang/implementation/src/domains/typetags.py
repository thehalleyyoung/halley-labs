"""
Type-tag lattice — finite powerset domain over Tag sort.

Tag = finite set of type-tag names representing the possible runtime types
a variable may have (int, str, float, list, dict, set, tuple, bool,
NoneType, bytes, and user-defined classes).

Supports subtype relationships (bool <: int, all <: object), MRO resolution,
attribute/method resolution, and compatibility checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    AbstractDomain,
    AbstractState,
    AbstractTransformer,
    AbstractValue,
    IRNode,
    WideningStrategy,
)


# ===================================================================
# TypeTag – individual type tag
# ===================================================================


@dataclass(frozen=True)
class TypeTag:
    """An individual type tag (e.g., 'int', 'str', 'MyClass')."""

    name: str
    module: Optional[str] = None
    bases: Tuple[str, ...] = ()  # direct base class names
    is_builtin: bool = True

    def __repr__(self) -> str:
        if self.module and not self.is_builtin:
            return f"Tag({self.module}.{self.name})"
        return f"Tag({self.name})"

    def qualified_name(self) -> str:
        if self.module and not self.is_builtin:
            return f"{self.module}.{self.name}"
        return self.name


# -- built-in tags -----------------------------------------------------------

BUILTIN_TAGS: Dict[str, TypeTag] = {}


def _make_builtin(name: str, bases: Tuple[str, ...] = ("object",)) -> TypeTag:
    tag = TypeTag(name=name, bases=bases, is_builtin=True)
    BUILTIN_TAGS[name] = tag
    return tag


TAG_OBJECT = _make_builtin("object", bases=())
TAG_INT = _make_builtin("int")
TAG_FLOAT = _make_builtin("float")
TAG_COMPLEX = _make_builtin("complex")
TAG_BOOL = _make_builtin("bool", bases=("int",))
TAG_STR = _make_builtin("str")
TAG_BYTES = _make_builtin("bytes")
TAG_LIST = _make_builtin("list")
TAG_TUPLE = _make_builtin("tuple")
TAG_SET = _make_builtin("set")
TAG_FROZENSET = _make_builtin("frozenset")
TAG_DICT = _make_builtin("dict")
TAG_NONETYPE = _make_builtin("NoneType")
TAG_TYPE = _make_builtin("type")
TAG_FUNCTION = _make_builtin("function")
TAG_MODULE = _make_builtin("module")
TAG_BYTEARRAY = _make_builtin("bytearray")
TAG_MEMORYVIEW = _make_builtin("memoryview")
TAG_RANGE = _make_builtin("range")
TAG_SLICE = _make_builtin("slice")
TAG_PROPERTY = _make_builtin("property")
TAG_STATICMETHOD = _make_builtin("staticmethod")
TAG_CLASSMETHOD = _make_builtin("classmethod")

ALL_BUILTIN_TAG_NAMES: FrozenSet[str] = frozenset(BUILTIN_TAGS.keys())


# ===================================================================
# Tag subtype hierarchy
# ===================================================================


class TypeHierarchy:
    """Manages the subtype hierarchy for type tags.

    Supports both built-in and user-defined class hierarchies.
    """

    def __init__(self) -> None:
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        for name, tag in BUILTIN_TAGS.items():
            self._parents[name] = list(tag.bases)
            for base in tag.bases:
                self._children.setdefault(base, []).append(name)

    def register_class(self, tag: TypeTag) -> None:
        self._parents[tag.name] = list(tag.bases)
        for base in tag.bases:
            self._children.setdefault(base, []).append(tag.name)

    def is_subtype(self, sub: str, sup: str) -> bool:
        """Return True if *sub* is a subtype of *sup*."""
        if sub == sup:
            return True
        if sup == "object":
            return True
        visited: Set[str] = set()
        worklist = [sub]
        while worklist:
            current = worklist.pop()
            if current in visited:
                continue
            visited.add(current)
            if current == sup:
                return True
            for parent in self._parents.get(current, []):
                worklist.append(parent)
        return False

    def subtypes_of(self, tag_name: str) -> FrozenSet[str]:
        """Return all known subtypes of *tag_name* (including itself)."""
        result: Set[str] = set()
        worklist = [tag_name]
        while worklist:
            current = worklist.pop()
            if current in result:
                continue
            result.add(current)
            for child in self._children.get(current, []):
                worklist.append(child)
        return frozenset(result)

    def supertypes_of(self, tag_name: str) -> FrozenSet[str]:
        """Return all known supertypes of *tag_name* (including itself)."""
        result: Set[str] = set()
        worklist = [tag_name]
        while worklist:
            current = worklist.pop()
            if current in result:
                continue
            result.add(current)
            for parent in self._parents.get(current, []):
                worklist.append(parent)
        return frozenset(result)

    def common_supertypes(self, tags: Iterable[str]) -> FrozenSet[str]:
        tag_list = list(tags)
        if not tag_list:
            return frozenset()
        result = self.supertypes_of(tag_list[0])
        for t in tag_list[1:]:
            result = result & self.supertypes_of(t)
        return result

    def direct_parents(self, tag_name: str) -> List[str]:
        return self._parents.get(tag_name, [])

    def direct_children(self, tag_name: str) -> List[str]:
        return self._children.get(tag_name, [])


_DEFAULT_HIERARCHY = TypeHierarchy()


# ===================================================================
# MROResolver – method resolution order
# ===================================================================


class MROResolver:
    """Compute C3-linearized Method Resolution Order for class hierarchies."""

    def __init__(self, hierarchy: Optional[TypeHierarchy] = None):
        self.hierarchy = hierarchy or _DEFAULT_HIERARCHY
        self._cache: Dict[str, List[str]] = {}

    def mro(self, class_name: str) -> List[str]:
        if class_name in self._cache:
            return self._cache[class_name]
        result = self._compute_mro(class_name)
        self._cache[class_name] = result
        return result

    def _compute_mro(self, class_name: str) -> List[str]:
        parents = self.hierarchy.direct_parents(class_name)
        if not parents:
            return [class_name]

        parent_mros = [self.mro(p) for p in parents]
        linearizations = parent_mros + [parents]
        result = [class_name]
        while linearizations:
            linearizations = [l for l in linearizations if l]
            if not linearizations:
                break
            candidate = None
            for lin in linearizations:
                head = lin[0]
                if all(head not in l[1:] for l in linearizations):
                    candidate = head
                    break
            if candidate is None:
                # Fall back to simple concatenation
                for lin in linearizations:
                    for item in lin:
                        if item not in result:
                            result.append(item)
                break
            result.append(candidate)
            linearizations = [
                [x for x in l if x != candidate] for l in linearizations
            ]
        if "object" not in result:
            result.append("object")
        return result


# ===================================================================
# TypeTagSet – set of possible type tags
# ===================================================================


@dataclass(frozen=True)
class TypeTagSet:
    """Set of possible type tags a variable may have.

    An empty set means ⊥ (unreachable).
    A set containing all tags means ⊤ (unknown type).
    """

    tags: FrozenSet[str]
    _is_top: bool = False

    @classmethod
    def top(cls) -> "TypeTagSet":
        return cls(tags=frozenset(), _is_top=True)

    @classmethod
    def bottom(cls) -> "TypeTagSet":
        return cls(tags=frozenset())

    @classmethod
    def singleton(cls, tag_name: str) -> "TypeTagSet":
        return cls(tags=frozenset({tag_name}))

    @classmethod
    def from_names(cls, *names: str) -> "TypeTagSet":
        return cls(tags=frozenset(names))

    @property
    def is_top(self) -> bool:
        return self._is_top

    @property
    def is_bottom(self) -> bool:
        return not self._is_top and len(self.tags) == 0

    @property
    def is_singleton(self) -> bool:
        return not self._is_top and len(self.tags) == 1

    def single_tag(self) -> Optional[str]:
        if self.is_singleton:
            return next(iter(self.tags))
        return None

    def contains(self, tag_name: str) -> bool:
        if self._is_top:
            return True
        return tag_name in self.tags

    def join(self, other: "TypeTagSet") -> "TypeTagSet":
        if self.is_top or other.is_top:
            return TypeTagSet.top()
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return TypeTagSet(tags=self.tags | other.tags)

    def meet(self, other: "TypeTagSet") -> "TypeTagSet":
        if self.is_top:
            return other
        if other.is_top:
            return self
        return TypeTagSet(tags=self.tags & other.tags)

    def remove(self, tag_name: str) -> "TypeTagSet":
        if self.is_top:
            return self
        return TypeTagSet(tags=self.tags - {tag_name})

    def add(self, tag_name: str) -> "TypeTagSet":
        if self.is_top:
            return self
        return TypeTagSet(tags=self.tags | {tag_name})

    def restrict_to(self, allowed: FrozenSet[str]) -> "TypeTagSet":
        if self.is_top:
            return TypeTagSet(tags=allowed)
        return TypeTagSet(tags=self.tags & allowed)

    def leq(self, other: "TypeTagSet") -> bool:
        if self.is_bottom:
            return True
        if other.is_top:
            return True
        if self.is_top:
            return False
        return self.tags <= other.tags

    def __repr__(self) -> str:
        if self.is_top:
            return "TypeTagSet(⊤)"
        if self.is_bottom:
            return "TypeTagSet(⊥)"
        return f"TypeTagSet({{{', '.join(sorted(self.tags))}}})"


# ===================================================================
# TypeTagValue – abstract value wrapping TypeTagSet
# ===================================================================


class TypeTagValue(AbstractValue):
    """Abstract value wrapping a TypeTagSet."""

    def __init__(self, tag_set: TypeTagSet):
        self.tag_set = tag_set

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeTagValue):
            return NotImplemented
        return self.tag_set == other.tag_set

    def __hash__(self) -> int:
        return hash(self.tag_set)

    def __repr__(self) -> str:
        return f"TypeTagValue({self.tag_set})"

    def is_bottom(self) -> bool:
        return self.tag_set.is_bottom

    def is_top(self) -> bool:
        return self.tag_set.is_top


# ===================================================================
# TypeTagDomain – AbstractDomain implementation
# ===================================================================


class TypeTagDomain(AbstractDomain[TypeTagValue]):
    """Abstract domain for type tags (finite powerset lattice)."""

    def __init__(
        self,
        hierarchy: Optional[TypeHierarchy] = None,
        max_set_size: int = 50,
    ):
        self.hierarchy = hierarchy or _DEFAULT_HIERARCHY
        self.max_set_size = max_set_size

    def top(self) -> TypeTagValue:
        return TypeTagValue(TypeTagSet.top())

    def bottom(self) -> TypeTagValue:
        return TypeTagValue(TypeTagSet.bottom())

    def join(self, a: TypeTagValue, b: TypeTagValue) -> TypeTagValue:
        result = a.tag_set.join(b.tag_set)
        if not result.is_top and len(result.tags) > self.max_set_size:
            return self.top()
        return TypeTagValue(result)

    def meet(self, a: TypeTagValue, b: TypeTagValue) -> TypeTagValue:
        return TypeTagValue(a.tag_set.meet(b.tag_set))

    def leq(self, a: TypeTagValue, b: TypeTagValue) -> bool:
        return a.tag_set.leq(b.tag_set)

    def widen(self, a: TypeTagValue, b: TypeTagValue) -> TypeTagValue:
        joined = a.tag_set.join(b.tag_set)
        if not joined.is_top and len(joined.tags) > self.max_set_size:
            return self.top()
        return TypeTagValue(joined)

    def narrow(self, a: TypeTagValue, b: TypeTagValue) -> TypeTagValue:
        return self.meet(a, b)

    def abstract(self, concrete: Any) -> TypeTagValue:
        tag_name = type(concrete).__name__
        return TypeTagValue(TypeTagSet.singleton(tag_name))

    def concretize(self, abstract_val: TypeTagValue) -> Any:
        if abstract_val.tag_set.is_bottom:
            return set()
        if abstract_val.tag_set.is_top:
            return "any type"
        return set(abstract_val.tag_set.tags)


# ===================================================================
# TypeTagWidening / Meet / Join
# ===================================================================


class TypeTagWidening(WideningStrategy[TypeTagValue]):
    """Widen type tag sets to ⊤ if they grow beyond a threshold."""

    def __init__(self, threshold: int = 10, delay: int = 0):
        self.threshold = threshold
        self.delay = delay

    def should_widen(self, node_id: int, iteration: int) -> bool:
        return iteration >= self.delay

    def apply(
        self,
        domain: AbstractDomain[TypeTagValue],
        old: TypeTagValue,
        new: TypeTagValue,
        iteration: int,
    ) -> TypeTagValue:
        joined = old.tag_set.join(new.tag_set)
        if not joined.is_top and len(joined.tags) > self.threshold:
            return TypeTagValue(TypeTagSet.top())
        return TypeTagValue(joined)


class TypeTagMeet:
    """Intersection of type tag sets."""

    @staticmethod
    def meet(a: TypeTagSet, b: TypeTagSet) -> TypeTagSet:
        return a.meet(b)

    @staticmethod
    def meet_values(a: TypeTagValue, b: TypeTagValue) -> TypeTagValue:
        return TypeTagValue(a.tag_set.meet(b.tag_set))


class TypeTagJoin:
    """Union of type tag sets."""

    @staticmethod
    def join(a: TypeTagSet, b: TypeTagSet) -> TypeTagSet:
        return a.join(b)

    @staticmethod
    def join_values(a: TypeTagValue, b: TypeTagValue) -> TypeTagValue:
        return TypeTagValue(a.tag_set.join(b.tag_set))


# ===================================================================
# TypeCompatibilityChecker
# ===================================================================


# Operation compatibility: which tags support which operations
_NUMERIC_OPS = frozenset({"+", "-", "*", "/", "//", "%", "**", "abs", "neg"})
_STRING_OPS = frozenset({"+", "*", "upper", "lower", "strip", "split", "join", "replace", "find", "startswith", "endswith", "format"})
_CONTAINER_OPS = frozenset({"len", "__getitem__", "__setitem__", "__delitem__", "__contains__", "append", "extend", "pop", "insert", "remove", "sort", "reverse"})
_DICT_OPS = frozenset({"keys", "values", "items", "get", "update", "pop", "setdefault", "__getitem__", "__setitem__", "__delitem__", "__contains__"})
_SET_OPS = frozenset({"add", "remove", "discard", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset"})

_TAG_OPERATIONS: Dict[str, FrozenSet[str]] = {
    "int": _NUMERIC_OPS | frozenset({"&", "|", "^", "<<", ">>", "~"}),
    "float": _NUMERIC_OPS,
    "complex": frozenset({"+", "-", "*", "/", "**", "abs"}),
    "bool": _NUMERIC_OPS | frozenset({"&", "|", "^", "~"}),
    "str": _STRING_OPS | frozenset({"len", "__getitem__", "__contains__"}),
    "bytes": frozenset({"len", "__getitem__", "__contains__", "+"}),
    "list": _CONTAINER_OPS | frozenset({"len"}),
    "tuple": frozenset({"len", "__getitem__", "__contains__", "+"}),
    "set": _SET_OPS | frozenset({"len", "__contains__"}),
    "frozenset": _SET_OPS | frozenset({"len", "__contains__"}) - frozenset({"add", "remove", "discard"}),
    "dict": _DICT_OPS | frozenset({"len", "__contains__"}),
    "NoneType": frozenset(),
    "range": frozenset({"len", "__getitem__", "__contains__"}),
}


class TypeCompatibilityChecker:
    """Check if an operation is valid for a given set of type tags."""

    def __init__(self, hierarchy: Optional[TypeHierarchy] = None):
        self.hierarchy = hierarchy or _DEFAULT_HIERARCHY

    def is_compatible(self, tag_set: TypeTagSet, operation: str) -> Optional[bool]:
        """Return True if all tags support the operation, False if none do, None if mixed."""
        if tag_set.is_top:
            return None
        if tag_set.is_bottom:
            return None

        supports = 0
        does_not_support = 0
        for tag_name in tag_set.tags:
            ops = _TAG_OPERATIONS.get(tag_name, frozenset())
            if operation in ops:
                supports += 1
            else:
                does_not_support += 1

        if does_not_support == 0:
            return True
        if supports == 0:
            return False
        return None

    def compatible_tags_for_op(self, operation: str) -> FrozenSet[str]:
        """Return the set of tags that support *operation*."""
        result: Set[str] = set()
        for tag_name, ops in _TAG_OPERATIONS.items():
            if operation in ops:
                result.add(tag_name)
        return frozenset(result)

    def possible_result_tags(self, operation: str, operand_tags: TypeTagSet) -> TypeTagSet:
        """Infer possible result type tags for an operation."""
        if operand_tags.is_bottom:
            return TypeTagSet.bottom()

        if operation in {"+", "-", "*", "//", "%", "**"}:
            if operand_tags.contains("float"):
                return TypeTagSet.from_names("int", "float")
            if operand_tags.contains("int") or operand_tags.contains("bool"):
                return TypeTagSet.singleton("int")
            if operand_tags.contains("str") and operation == "+":
                return TypeTagSet.singleton("str")
            if operand_tags.contains("list") and operation == "+":
                return TypeTagSet.singleton("list")
            if operand_tags.contains("str") and operation == "*":
                return TypeTagSet.singleton("str")
            if operand_tags.contains("list") and operation == "*":
                return TypeTagSet.singleton("list")
            return TypeTagSet.top()

        if operation in {"<", "<=", ">", ">=", "==", "!=", "is", "is not", "in", "not in"}:
            return TypeTagSet.singleton("bool")

        if operation == "len":
            return TypeTagSet.singleton("int")

        if operation in {"not", "and", "or"}:
            return TypeTagSet.singleton("bool")

        return TypeTagSet.top()


# ===================================================================
# AttributeResolver
# ===================================================================


_BUILTIN_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    "int": frozenset({"real", "imag", "numerator", "denominator", "bit_length", "bit_count", "to_bytes", "from_bytes", "conjugate"}),
    "float": frozenset({"real", "imag", "is_integer", "hex", "fromhex", "conjugate", "as_integer_ratio"}),
    "str": frozenset({"upper", "lower", "strip", "lstrip", "rstrip", "split", "rsplit", "join", "replace", "find", "rfind", "index", "rindex", "count", "startswith", "endswith", "format", "format_map", "encode", "isdigit", "isalpha", "isalnum", "isspace", "isupper", "islower", "title", "capitalize", "center", "ljust", "rjust", "zfill", "expandtabs", "partition", "rpartition", "maketrans", "translate", "removeprefix", "removesuffix"}),
    "bytes": frozenset({"decode", "hex", "count", "find", "rfind", "index", "rindex", "replace", "split", "rsplit", "join", "strip", "lstrip", "rstrip", "startswith", "endswith", "upper", "lower", "isdigit", "isalpha", "isalnum", "isspace"}),
    "list": frozenset({"append", "extend", "insert", "remove", "pop", "clear", "index", "count", "sort", "reverse", "copy"}),
    "tuple": frozenset({"count", "index"}),
    "set": frozenset({"add", "remove", "discard", "pop", "clear", "copy", "union", "intersection", "difference", "symmetric_difference", "update", "intersection_update", "difference_update", "symmetric_difference_update", "issubset", "issuperset", "isdisjoint"}),
    "frozenset": frozenset({"copy", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint"}),
    "dict": frozenset({"keys", "values", "items", "get", "pop", "popitem", "setdefault", "update", "clear", "copy", "fromkeys"}),
    "NoneType": frozenset(),
    "bool": frozenset({"real", "imag", "numerator", "denominator", "bit_length", "conjugate"}),
}


class AttributeResolver:
    """Resolve available attributes for a type tag set."""

    def __init__(self, hierarchy: Optional[TypeHierarchy] = None):
        self.hierarchy = hierarchy or _DEFAULT_HIERARCHY
        self._custom_attrs: Dict[str, Set[str]] = {}

    def register_attributes(self, tag_name: str, attrs: Iterable[str]) -> None:
        self._custom_attrs.setdefault(tag_name, set()).update(attrs)

    def available_attributes(self, tag_set: TypeTagSet) -> FrozenSet[str]:
        """Return attributes available on all tags in the set (intersection)."""
        if tag_set.is_top or tag_set.is_bottom:
            return frozenset()
        result: Optional[Set[str]] = None
        for tag_name in tag_set.tags:
            attrs = set(_BUILTIN_ATTRIBUTES.get(tag_name, frozenset()))
            attrs.update(self._custom_attrs.get(tag_name, set()))
            if result is None:
                result = attrs
            else:
                result &= attrs
        return frozenset(result) if result is not None else frozenset()

    def has_attribute(self, tag_set: TypeTagSet, attr: str) -> Optional[bool]:
        """Check if the tag set definitely/maybe/never has an attribute."""
        if tag_set.is_top:
            return None
        if tag_set.is_bottom:
            return None
        has_it = 0
        no = 0
        for tag_name in tag_set.tags:
            attrs = set(_BUILTIN_ATTRIBUTES.get(tag_name, frozenset()))
            attrs.update(self._custom_attrs.get(tag_name, set()))
            if attr in attrs:
                has_it += 1
            else:
                no += 1
        if no == 0:
            return True
        if has_it == 0:
            return False
        return None

    def tags_with_attribute(self, tag_set: TypeTagSet, attr: str) -> TypeTagSet:
        """Narrow tag set to only tags that have the given attribute."""
        if tag_set.is_top:
            matching: Set[str] = set()
            for name, attrs in _BUILTIN_ATTRIBUTES.items():
                if attr in attrs:
                    matching.add(name)
            for name, attrs in self._custom_attrs.items():
                if attr in attrs:
                    matching.add(name)
            return TypeTagSet(tags=frozenset(matching))
        if tag_set.is_bottom:
            return tag_set
        matching_tags: Set[str] = set()
        for tag_name in tag_set.tags:
            attrs = set(_BUILTIN_ATTRIBUTES.get(tag_name, frozenset()))
            attrs.update(self._custom_attrs.get(tag_name, set()))
            if attr in attrs:
                matching_tags.add(tag_name)
        return TypeTagSet(tags=frozenset(matching_tags))


# ===================================================================
# MethodResolver
# ===================================================================


class MethodResolver:
    """Resolve available methods for a type tag set using MRO."""

    def __init__(
        self,
        hierarchy: Optional[TypeHierarchy] = None,
        mro_resolver: Optional[MROResolver] = None,
    ):
        self.hierarchy = hierarchy or _DEFAULT_HIERARCHY
        self.mro_resolver = mro_resolver or MROResolver(self.hierarchy)
        self.attr_resolver = AttributeResolver(self.hierarchy)
        self._method_signatures: Dict[Tuple[str, str], "MethodSignature"] = {}

    def register_method(
        self,
        tag_name: str,
        method_name: str,
        param_count: int = 0,
        return_tag: Optional[str] = None,
    ) -> None:
        self._method_signatures[(tag_name, method_name)] = MethodSignature(
            name=method_name,
            owner=tag_name,
            param_count=param_count,
            return_tag=return_tag,
        )

    def resolve_method(
        self, tag_set: TypeTagSet, method_name: str
    ) -> Optional["MethodResolution"]:
        """Resolve a method call on a tag set."""
        if tag_set.is_bottom:
            return None

        possible_owners: List[str] = []
        return_tags: Set[str] = set()

        tags_to_check = tag_set.tags if not tag_set.is_top else frozenset(BUILTIN_TAGS.keys())
        for tag_name in tags_to_check:
            mro = self.mro_resolver.mro(tag_name)
            for cls_name in mro:
                attrs = set(_BUILTIN_ATTRIBUTES.get(cls_name, frozenset()))
                if method_name in attrs:
                    possible_owners.append(tag_name)
                    sig = self._method_signatures.get((cls_name, method_name))
                    if sig and sig.return_tag:
                        return_tags.add(sig.return_tag)
                    break

        if not possible_owners:
            return None

        return MethodResolution(
            method_name=method_name,
            possible_owners=frozenset(possible_owners),
            return_tags=TypeTagSet(tags=frozenset(return_tags)) if return_tags else TypeTagSet.top(),
            is_definite=len(possible_owners) == len(tags_to_check),
        )


@dataclass(frozen=True)
class MethodSignature:
    name: str
    owner: str
    param_count: int = 0
    return_tag: Optional[str] = None


@dataclass(frozen=True)
class MethodResolution:
    method_name: str
    possible_owners: FrozenSet[str]
    return_tags: TypeTagSet
    is_definite: bool


# ===================================================================
# TypeTagSerializer
# ===================================================================


class TypeTagSerializer:
    """Serialize/deserialize TypeTagSets for caching."""

    @staticmethod
    def serialize(tag_set: TypeTagSet) -> str:
        if tag_set.is_top:
            return "⊤"
        if tag_set.is_bottom:
            return "⊥"
        return ",".join(sorted(tag_set.tags))

    @staticmethod
    def deserialize(s: str) -> TypeTagSet:
        if s == "⊤":
            return TypeTagSet.top()
        if s == "⊥":
            return TypeTagSet.bottom()
        if not s:
            return TypeTagSet.bottom()
        return TypeTagSet(tags=frozenset(s.split(",")))

    @staticmethod
    def serialize_value(val: TypeTagValue) -> str:
        return TypeTagSerializer.serialize(val.tag_set)

    @staticmethod
    def deserialize_value(s: str) -> TypeTagValue:
        return TypeTagValue(TypeTagSerializer.deserialize(s))


# ===================================================================
# TypeTag abstract transformers
# ===================================================================


class TypeTagTransformer(AbstractTransformer[TypeTagValue]):
    """Abstract transformer for type-tag domain operations."""

    def __init__(
        self,
        domain: TypeTagDomain,
        hierarchy: Optional[TypeHierarchy] = None,
    ):
        self.domain = domain
        self.hierarchy = hierarchy or _DEFAULT_HIERARCHY
        self.compat_checker = TypeCompatibilityChecker(self.hierarchy)

    def assign(
        self, state: AbstractState[TypeTagValue], var: str, expr: Any
    ) -> AbstractState[TypeTagValue]:
        val = self._eval_expr(state, expr)
        return state.set(var, val)

    def guard(
        self, state: AbstractState[TypeTagValue], condition: Any, branch: bool
    ) -> AbstractState[TypeTagValue]:
        if not isinstance(condition, (list, tuple)):
            return state

        if len(condition) == 3:
            op, arg1, arg2 = condition
            if op == "isinstance":
                return self._guard_isinstance(state, arg1, arg2, branch)
            if op == "is" and arg2 == "None":
                return self._guard_is_none(state, arg1, branch)
            if op == "is not" and arg2 == "None":
                return self._guard_is_none(state, arg1, not branch)
        elif len(condition) == 2:
            op, arg = condition
            if op == "type":
                return self._guard_type_check(state, arg, branch)

        return state

    def call(
        self,
        state: AbstractState[TypeTagValue],
        func: str,
        args: List[Any],
        result_var: Optional[str] = None,
    ) -> AbstractState[TypeTagValue]:
        result = self._eval_call(state, func, args)
        if result_var is not None:
            return state.set(result_var, result)
        return state

    # -- isinstance guard ----------------------------------------------------

    def _guard_isinstance(
        self,
        state: AbstractState[TypeTagValue],
        var: str,
        type_name: Any,
        branch: bool,
    ) -> AbstractState[TypeTagValue]:
        current = state.get(var)
        if current is None:
            return state

        if isinstance(type_name, str):
            target_tags = self.hierarchy.subtypes_of(type_name)
        elif isinstance(type_name, (list, tuple)):
            target_tags: FrozenSet[str] = frozenset()
            for tn in type_name:
                if isinstance(tn, str):
                    target_tags = target_tags | self.hierarchy.subtypes_of(tn)
        else:
            return state

        if branch:
            # True branch: narrow to target tags
            if current.tag_set.is_top:
                return state.set(var, TypeTagValue(TypeTagSet(tags=target_tags)))
            new_tags = current.tag_set.tags & target_tags
            return state.set(var, TypeTagValue(TypeTagSet(tags=new_tags)))
        else:
            # False branch: remove target tags
            if current.tag_set.is_top:
                return state
            new_tags = current.tag_set.tags - target_tags
            return state.set(var, TypeTagValue(TypeTagSet(tags=new_tags)))

    # -- is None guard -------------------------------------------------------

    def _guard_is_none(
        self,
        state: AbstractState[TypeTagValue],
        var: str,
        branch: bool,
    ) -> AbstractState[TypeTagValue]:
        current = state.get(var)
        if current is None:
            return state

        if branch:
            return state.set(var, TypeTagValue(TypeTagSet.singleton("NoneType")))
        else:
            new_tags = current.tag_set.remove("NoneType")
            return state.set(var, TypeTagValue(new_tags))

    # -- type check guard ----------------------------------------------------

    def _guard_type_check(
        self,
        state: AbstractState[TypeTagValue],
        check: Any,
        branch: bool,
    ) -> AbstractState[TypeTagValue]:
        if isinstance(check, (list, tuple)) and len(check) == 2:
            var, type_name = check
            if isinstance(var, str) and isinstance(type_name, str):
                return self._guard_isinstance(state, var, type_name, branch)
        return state

    # -- expression evaluation -----------------------------------------------

    def _eval_expr(self, state: AbstractState[TypeTagValue], expr: Any) -> TypeTagValue:
        if expr is None:
            return TypeTagValue(TypeTagSet.singleton("NoneType"))
        if isinstance(expr, bool):
            return TypeTagValue(TypeTagSet.singleton("bool"))
        if isinstance(expr, int):
            return TypeTagValue(TypeTagSet.singleton("int"))
        if isinstance(expr, float):
            return TypeTagValue(TypeTagSet.singleton("float"))
        if isinstance(expr, str):
            val = state.get(expr)
            if val is not None:
                return val
            return TypeTagValue(TypeTagSet.singleton("str"))
        if isinstance(expr, (list, tuple)):
            if len(expr) >= 2:
                op = expr[0]
                if op in {"+", "-", "*", "//", "%", "**"}:
                    return self._eval_arith_result(state, op, expr[1:])
                if op in {"<", "<=", ">", ">=", "==", "!=", "is", "is not", "in", "not in", "and", "or", "not"}:
                    return TypeTagValue(TypeTagSet.singleton("bool"))
                if op == "constructor":
                    if len(expr) >= 2 and isinstance(expr[1], str):
                        return TypeTagValue(TypeTagSet.singleton(expr[1]))
            return self.domain.top()
        return self.domain.top()

    def _eval_arith_result(
        self, state: AbstractState[TypeTagValue], op: str, operands: List[Any]
    ) -> TypeTagValue:
        operand_tags: Set[str] = set()
        for operand in operands:
            val = self._eval_expr(state, operand)
            if val.tag_set.is_top:
                return self.domain.top()
            operand_tags.update(val.tag_set.tags)

        if "float" in operand_tags:
            return TypeTagValue(TypeTagSet.singleton("float"))
        if "complex" in operand_tags:
            return TypeTagValue(TypeTagSet.singleton("complex"))

        if operand_tags <= {"int", "bool"}:
            return TypeTagValue(TypeTagSet.singleton("int"))

        if op == "+" and "str" in operand_tags:
            return TypeTagValue(TypeTagSet.singleton("str"))
        if op == "+" and "list" in operand_tags:
            return TypeTagValue(TypeTagSet.singleton("list"))
        if op == "+" and "tuple" in operand_tags:
            return TypeTagValue(TypeTagSet.singleton("tuple"))

        if op == "*":
            if "str" in operand_tags and ("int" in operand_tags or "bool" in operand_tags):
                return TypeTagValue(TypeTagSet.singleton("str"))
            if "list" in operand_tags and ("int" in operand_tags or "bool" in operand_tags):
                return TypeTagValue(TypeTagSet.singleton("list"))

        return self.domain.top()

    def _eval_call(
        self,
        state: AbstractState[TypeTagValue],
        func: str,
        args: List[Any],
    ) -> TypeTagValue:
        _FUNC_RETURN_TAGS: Dict[str, str] = {
            "int": "int",
            "float": "float",
            "str": "str",
            "bool": "bool",
            "list": "list",
            "tuple": "tuple",
            "set": "set",
            "frozenset": "frozenset",
            "dict": "dict",
            "bytes": "bytes",
            "bytearray": "bytearray",
            "len": "int",
            "abs": "int",
            "min": "int",
            "max": "int",
            "sum": "int",
            "round": "int",
            "range": "range",
            "enumerate": "list",
            "zip": "list",
            "map": "list",
            "filter": "list",
            "sorted": "list",
            "reversed": "list",
            "isinstance": "bool",
            "issubclass": "bool",
            "hasattr": "bool",
            "callable": "bool",
            "id": "int",
            "hash": "int",
            "repr": "str",
            "type": "type",
            "input": "str",
            "ord": "int",
            "chr": "str",
            "hex": "str",
            "oct": "str",
            "bin": "str",
        }
        tag = _FUNC_RETURN_TAGS.get(func)
        if tag is not None:
            return TypeTagValue(TypeTagSet.singleton(tag))
        return self.domain.top()
