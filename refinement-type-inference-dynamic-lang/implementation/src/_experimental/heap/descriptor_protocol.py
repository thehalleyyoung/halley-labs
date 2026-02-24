"""Modeling of Python's descriptor protocol for refinement type inference.

Implements the full attribute lookup semantics including data descriptors,
non-data descriptors, property/classmethod/staticmethod analysis, __slots__,
and MRO-based resolution as specified in the Python data model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from .class_hierarchy import (
    ClassHierarchyAnalyzer,
    ClassInfo,
    MethodInfo,
    ParamInfo,
    PropertyInfo,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DescriptorKind(Enum):
    """Classification of an attribute with respect to the descriptor protocol."""
    DATA = auto()
    NON_DATA = auto()
    PLAIN = auto()
    PROPERTY = auto()
    CLASSMETHOD = auto()
    STATICMETHOD = auto()
    SLOT = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DescriptorInfo:
    """Information about a single descriptor (or plain attribute)."""
    kind: DescriptorKind
    defining_class: str
    has_get: bool
    has_set: bool
    has_delete: bool
    attr_name: str

    @property
    def is_data(self) -> bool:
        return self.has_get and self.has_set

    @property
    def is_non_data(self) -> bool:
        return self.has_get and not self.has_set


@dataclass
class AttributeResolution:
    """Result of resolving an attribute access on an object or class."""
    source: str  # 'instance_dict' | 'data_descriptor' | 'non_data_descriptor'
                 # | 'class_var' | 'getattr' | 'getattribute' | 'error'
    defining_class: Optional[str] = None
    descriptor_info: Optional[DescriptorInfo] = None
    value_type: Optional[str] = None
    is_writable: bool = False


@dataclass
class PropertyType:
    """Refined type information for a ``@property``."""
    name: str
    getter_type: Optional[str] = None
    setter_type: Optional[str] = None
    deleter_exists: bool = False
    owner_class: str = ""


@dataclass
class ClassMethodType:
    """Refined type information for a ``@classmethod``."""
    name: str
    params: List[ParamInfo] = field(default_factory=list)
    return_type: Optional[str] = None
    owner_class: str = ""


@dataclass
class StaticMethodType:
    """Refined type information for a ``@staticmethod``."""
    name: str
    params: List[ParamInfo] = field(default_factory=list)
    return_type: Optional[str] = None
    owner_class: str = ""


@dataclass
class SlotInfo:
    """Information about a single ``__slots__`` entry."""
    name: str
    annotation: Optional[str] = None
    has_default: bool = False


# ---------------------------------------------------------------------------
# Built-in descriptor knowledge
# ---------------------------------------------------------------------------

# Builtin types that are known data descriptors (have both __get__ and __set__)
_BUILTIN_DATA_DESCRIPTORS: Dict[str, Set[str]] = {
    "object": set(),
    "type": {"__dict__", "__bases__", "__name__", "__qualname__",
             "__module__", "__abstractmethods__", "__subclasshook__"},
    "property": {"fget", "fset", "fdel", "__doc__"},
    "classmethod": set(),
    "staticmethod": set(),
    "super": set(),
    "function": {"__dict__", "__code__", "__globals__", "__name__",
                 "__defaults__", "__kwdefaults__", "__annotations__"},
}

# Builtin types that are known non-data descriptors (only __get__)
_BUILTIN_NON_DATA_DESCRIPTORS: Dict[str, Set[str]] = {
    "function": set(),  # functions are non-data descriptors themselves
    "classmethod_descriptor": set(),
    "staticmethod_descriptor": set(),
}

# Dunders that signal descriptor-ness
_DESCRIPTOR_DUNDERS: Set[str] = {"__get__", "__set__", "__delete__"}


# ---------------------------------------------------------------------------
# DescriptorAnalyzer
# ---------------------------------------------------------------------------

class DescriptorAnalyzer:
    """Analyses and resolves attribute accesses using Python's descriptor protocol.

    The analyzer faithfully models the CPython attribute lookup order:
      1. Data descriptors found on ``type(obj).__mro__``
      2. Instance ``__dict__``
      3. Non-data descriptors found on ``type(obj).__mro__``
    with additional handling for ``__getattr__`` / ``__getattribute__``
    overrides, ``__slots__``, and the special decorator descriptors
    (``property``, ``classmethod``, ``staticmethod``).
    """

    def __init__(self) -> None:
        # Cache of already-classified descriptors keyed by (class, attr).
        self._descriptor_cache: Dict[Tuple[str, str], DescriptorInfo] = {}

        # Pre-populate knowledge of Python builtins.
        self._known_data_descriptors: Dict[str, Set[str]] = dict(
            _BUILTIN_DATA_DESCRIPTORS
        )
        self._known_non_data_descriptors: Dict[str, Set[str]] = dict(
            _BUILTIN_NON_DATA_DESCRIPTORS
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_descriptor(
        self,
        class_name: str,
        attr_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> DescriptorInfo:
        """Classify *attr_name* defined on *class_name* as a descriptor kind.

        The classification checks whether the attribute's own type provides
        ``__get__``, ``__set__``, or ``__delete__`` and maps to the
        appropriate :class:`DescriptorKind`.
        """
        cache_key = (class_name, attr_name)
        if cache_key in self._descriptor_cache:
            return self._descriptor_cache[cache_key]

        info = self._do_classify(class_name, attr_name, hierarchy)
        self._descriptor_cache[cache_key] = info
        return info

    def resolve_attribute_access(
        self,
        obj_type: str,
        attr_name: str,
        hierarchy: ClassHierarchyAnalyzer,
        is_class_access: bool = False,
    ) -> AttributeResolution:
        """Resolve *attr_name* on an instance of *obj_type*.

        Follows the standard Python attribute lookup order:

        1. ``type(obj).__mro__`` is searched for **data descriptors**.
        2. ``obj.__dict__`` (instance dictionary) is checked.
        3. ``type(obj).__mro__`` is searched for **non-data descriptors**.
        4. ``__getattr__`` fallback.
        5. Error.

        When *is_class_access* is ``True`` the lookup is performed as if
        the access is on the class itself (i.e. ``Cls.attr``).
        """
        if is_class_access:
            return self.resolve_class_attribute(obj_type, attr_name, hierarchy)

        # Check for __getattribute__ override first – if present the class
        # may implement entirely custom lookup logic.
        if self._has_custom_getattribute(obj_type, hierarchy):
            return AttributeResolution(
                source="getattribute",
                defining_class=obj_type,
                value_type=None,
                is_writable=False,
            )

        # Step 1 – look for data descriptors in the MRO.
        mro_hit = self._find_in_mro(obj_type, attr_name, hierarchy)
        if mro_hit is not None:
            owner_class, desc_info = mro_hit
            if self._is_data_descriptor(desc_info):
                return AttributeResolution(
                    source="data_descriptor",
                    defining_class=owner_class,
                    descriptor_info=desc_info,
                    value_type=self._infer_descriptor_value_type(
                        desc_info, hierarchy
                    ),
                    is_writable=desc_info.has_set,
                )

        # Step 2 – check instance __dict__.
        class_info = hierarchy.get_class(obj_type)
        if class_info is not None:
            has_slots = self._class_uses_slots(class_info)
            if has_slots:
                slot_names = self._collect_slot_names(obj_type, hierarchy)
                if attr_name in slot_names:
                    return AttributeResolution(
                        source="instance_dict",
                        defining_class=obj_type,
                        value_type=self._slot_annotation(
                            obj_type, attr_name, hierarchy
                        ),
                        is_writable=True,
                    )
            else:
                # Without __slots__ any instance attribute is possible.
                instance_attrs = self._collect_instance_attrs(
                    obj_type, hierarchy
                )
                if attr_name in instance_attrs:
                    return AttributeResolution(
                        source="instance_dict",
                        defining_class=obj_type,
                        value_type=instance_attrs.get(attr_name),
                        is_writable=True,
                    )

        # Step 3 – non-data descriptors / plain class variables.
        if mro_hit is not None:
            owner_class, desc_info = mro_hit
            if self._is_non_data_descriptor(desc_info):
                return AttributeResolution(
                    source="non_data_descriptor",
                    defining_class=owner_class,
                    descriptor_info=desc_info,
                    value_type=self._infer_descriptor_value_type(
                        desc_info, hierarchy
                    ),
                    is_writable=False,
                )
            # Plain class variable (no descriptor protocol methods at all).
            return AttributeResolution(
                source="class_var",
                defining_class=owner_class,
                descriptor_info=desc_info,
                value_type=self._infer_descriptor_value_type(
                    desc_info, hierarchy
                ),
                is_writable=True,
            )

        # Step 4 – __getattr__ fallback.
        if self._has_dunder(obj_type, "__getattr__", hierarchy):
            return AttributeResolution(
                source="getattr",
                defining_class=obj_type,
                value_type=None,
                is_writable=False,
            )

        # Step 5 – not found.
        return AttributeResolution(source="error")

    def resolve_class_attribute(
        self,
        class_name: str,
        attr_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> AttributeResolution:
        """Resolve *attr_name* when accessed on the **class object** itself.

        For class-level access the MRO of the *metaclass* (usually ``type``)
        governs data-descriptor priority, but in practice we search the
        class's own MRO for descriptors and class variables.
        """
        # Check the metaclass MRO for data descriptors.  In typical code
        # this means ``type.__dict__`` attributes (e.g. ``__dict__``,
        # ``__subclasshook__``).
        meta_hit = self._find_in_mro("type", attr_name, hierarchy)
        if meta_hit is not None:
            _, meta_desc = meta_hit
            if self._is_data_descriptor(meta_desc):
                return AttributeResolution(
                    source="data_descriptor",
                    defining_class=meta_hit[0],
                    descriptor_info=meta_desc,
                    value_type=None,
                    is_writable=meta_desc.has_set,
                )

        # Search the class's own MRO.
        mro_hit = self._find_in_mro(class_name, attr_name, hierarchy)
        if mro_hit is not None:
            owner, desc_info = mro_hit
            if desc_info.kind == DescriptorKind.CLASSMETHOD:
                return AttributeResolution(
                    source="data_descriptor",
                    defining_class=owner,
                    descriptor_info=desc_info,
                    value_type=self._infer_descriptor_value_type(
                        desc_info, hierarchy
                    ),
                    is_writable=False,
                )
            if desc_info.kind == DescriptorKind.STATICMETHOD:
                return AttributeResolution(
                    source="non_data_descriptor",
                    defining_class=owner,
                    descriptor_info=desc_info,
                    value_type=self._infer_descriptor_value_type(
                        desc_info, hierarchy
                    ),
                    is_writable=False,
                )
            if self._is_data_descriptor(desc_info):
                return AttributeResolution(
                    source="data_descriptor",
                    defining_class=owner,
                    descriptor_info=desc_info,
                    value_type=self._infer_descriptor_value_type(
                        desc_info, hierarchy
                    ),
                    is_writable=desc_info.has_set,
                )
            if self._is_non_data_descriptor(desc_info):
                return AttributeResolution(
                    source="non_data_descriptor",
                    defining_class=owner,
                    descriptor_info=desc_info,
                    value_type=self._infer_descriptor_value_type(
                        desc_info, hierarchy
                    ),
                    is_writable=False,
                )
            # Plain class variable.
            return AttributeResolution(
                source="class_var",
                defining_class=owner,
                descriptor_info=desc_info,
                value_type=self._infer_descriptor_value_type(
                    desc_info, hierarchy
                ),
                is_writable=True,
            )

        # Metaclass non-data descriptors.
        if meta_hit is not None:
            _, meta_desc = meta_hit
            return AttributeResolution(
                source="non_data_descriptor",
                defining_class=meta_hit[0],
                descriptor_info=meta_desc,
                value_type=None,
                is_writable=False,
            )

        # __getattr__ on the metaclass.
        if self._has_dunder("type", "__getattr__", hierarchy):
            return AttributeResolution(
                source="getattr",
                defining_class="type",
                value_type=None,
                is_writable=False,
            )

        return AttributeResolution(source="error")

    # -- Specialized analyzers ---------------------------------------------

    def analyze_property(
        self,
        class_name: str,
        prop_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> PropertyType:
        """Return refined type information for a ``@property``."""
        class_info = hierarchy.get_class(class_name)
        if class_info is None:
            return PropertyType(name=prop_name, owner_class=class_name)

        getter_type: Optional[str] = None
        setter_type: Optional[str] = None
        deleter_exists: bool = False

        # Look for a PropertyInfo on the class.
        prop_info = self._find_property_info(class_info, prop_name)
        if prop_info is not None:
            getter_type = getattr(prop_info, "return_type", None)
            setter_type = getattr(prop_info, "setter_type", None)
            deleter_exists = getattr(prop_info, "has_deleter", False)
        else:
            # Fall back: look for getter/setter/deleter methods by convention.
            getter = self._find_method(class_info, prop_name)
            if getter is not None:
                getter_type = getattr(getter, "return_type", None)
            setter = self._find_method(class_info, prop_name + ".setter")
            if setter is None:
                setter = self._find_method(class_info, f"_{prop_name}_setter")
            if setter is not None:
                setter_params = getattr(setter, "params", [])
                if len(setter_params) >= 2:
                    setter_type = getattr(setter_params[1], "annotation", None)
            deleter = self._find_method(class_info, prop_name + ".deleter")
            if deleter is None:
                deleter = self._find_method(
                    class_info, f"_{prop_name}_deleter"
                )
            deleter_exists = deleter is not None

        # Walk the MRO to find inherited property facets.
        if getter_type is None or setter_type is None:
            for base_name in self._iter_mro(class_name, hierarchy):
                if base_name == class_name:
                    continue
                base_info = hierarchy.get_class(base_name)
                if base_info is None:
                    continue
                base_prop = self._find_property_info(base_info, prop_name)
                if base_prop is not None:
                    if getter_type is None:
                        getter_type = getattr(base_prop, "return_type", None)
                    if setter_type is None:
                        setter_type = getattr(base_prop, "setter_type", None)
                    if not deleter_exists:
                        deleter_exists = getattr(
                            base_prop, "has_deleter", False
                        )
                    break

        return PropertyType(
            name=prop_name,
            getter_type=getter_type,
            setter_type=setter_type,
            deleter_exists=deleter_exists,
            owner_class=class_name,
        )

    def analyze_classmethod(
        self,
        class_name: str,
        method_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> ClassMethodType:
        """Return refined type information for a ``@classmethod``."""
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            method = self._find_method(cls_info, method_name)
            if method is not None:
                params = list(getattr(method, "params", []))
                return_type = getattr(method, "return_type", None)
                # Drop the implicit ``cls`` parameter for the external view.
                external_params = params[1:] if params else []
                return ClassMethodType(
                    name=method_name,
                    params=external_params,
                    return_type=return_type,
                    owner_class=owner,
                )
        return ClassMethodType(name=method_name, owner_class=class_name)

    def analyze_staticmethod(
        self,
        class_name: str,
        method_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> StaticMethodType:
        """Return refined type information for a ``@staticmethod``."""
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            method = self._find_method(cls_info, method_name)
            if method is not None:
                params = list(getattr(method, "params", []))
                return_type = getattr(method, "return_type", None)
                return StaticMethodType(
                    name=method_name,
                    params=params,
                    return_type=return_type,
                    owner_class=owner,
                )
        return StaticMethodType(name=method_name, owner_class=class_name)

    def analyze_slots(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> List[SlotInfo]:
        """Collect ``__slots__`` entries for *class_name* and its bases."""
        result: List[SlotInfo] = []
        seen: Set[str] = set()

        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            if not self._class_uses_slots(cls_info):
                continue
            raw_slots = self._extract_raw_slots(cls_info)
            for slot_name, annotation in raw_slots:
                if slot_name in seen:
                    continue
                seen.add(slot_name)
                has_default = self._slot_has_default(
                    cls_info, slot_name
                )
                result.append(
                    SlotInfo(
                        name=slot_name,
                        annotation=annotation,
                        has_default=has_default,
                    )
                )
        return result

    def check_descriptor_consistency(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> List[str]:
        """Return a list of diagnostic messages about descriptor issues.

        Checks performed:
        * Data descriptor in a base shadowed by a plain attribute in a subclass.
        * ``__slots__`` entry that conflicts with a class variable.
        * Property without a getter.
        * Classmethod/staticmethod overriding a data descriptor.
        * Inconsistent ``__slots__`` across the MRO (some bases with, some
          without).
        """
        issues: List[str] = []
        cls_info = hierarchy.get_class(class_name)
        if cls_info is None:
            return issues

        # Collect all attribute names visible in the MRO.
        all_attrs = self._collect_all_mro_attrs(class_name, hierarchy)

        for attr_name in all_attrs:
            desc = self.classify_descriptor(class_name, attr_name, hierarchy)

            # Shadow check: data descriptor in base, plain/non-data in class.
            if desc.defining_class != class_name:
                local = self._find_local_attr(cls_info, attr_name)
                if local is not None:
                    if self._is_data_descriptor(desc) and not self._is_data_descriptor(local):
                        issues.append(
                            f"'{attr_name}': data descriptor from "
                            f"'{desc.defining_class}' is shadowed by a "
                            f"plain attribute in '{class_name}'"
                        )

            # Property without getter.
            if desc.kind == DescriptorKind.PROPERTY:
                prop = self.analyze_property(
                    class_name, attr_name, hierarchy
                )
                if prop.getter_type is None:
                    prop_info = self._find_property_info(cls_info, attr_name)
                    if prop_info is not None and not getattr(
                        prop_info, "has_getter", True
                    ):
                        issues.append(
                            f"'{attr_name}': property on '{class_name}' "
                            "has no getter"
                        )

        # Slot / class-var conflicts.
        if self._class_uses_slots(cls_info):
            slot_names = self._collect_slot_names(class_name, hierarchy)
            class_vars = self._collect_class_vars(cls_info)
            for s in slot_names:
                if s in class_vars:
                    issues.append(
                        f"'{s}': __slots__ entry conflicts with a class "
                        f"variable in '{class_name}'"
                    )

        # Inconsistent __slots__ in MRO.
        mro = list(self._iter_mro(class_name, hierarchy))
        slots_status: List[Tuple[str, bool]] = []
        for base in mro:
            base_info = hierarchy.get_class(base)
            if base_info is not None:
                slots_status.append(
                    (base, self._class_uses_slots(base_info))
                )
        has_any_slots = any(s for _, s in slots_status)
        has_any_no_slots = any(
            not s for name, s in slots_status if name != "object"
        )
        if has_any_slots and has_any_no_slots:
            issues.append(
                f"Inconsistent __slots__ usage in MRO of '{class_name}': "
                "some bases define __slots__ and some do not"
            )

        return issues

    def get_settable_attributes(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Dict[str, AttributeResolution]:
        """Return all attributes that can be set on instances of *class_name*."""
        result: Dict[str, AttributeResolution] = {}
        all_attrs = self._collect_all_mro_attrs(class_name, hierarchy)

        # Instance attributes from __init__ or __slots__.
        instance_attrs = self._collect_instance_attrs(class_name, hierarchy)
        for attr in instance_attrs:
            all_attrs.add(attr)

        slot_names = self._collect_slot_names(class_name, hierarchy)
        for s in slot_names:
            all_attrs.add(s)

        for attr_name in all_attrs:
            resolution = self.resolve_attribute_access(
                class_name, attr_name, hierarchy
            )
            if resolution.is_writable:
                result[attr_name] = resolution
        return result

    def get_deletable_attributes(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Dict[str, AttributeResolution]:
        """Return all attributes that can be deleted on instances of *class_name*."""
        result: Dict[str, AttributeResolution] = {}
        all_attrs = self._collect_all_mro_attrs(class_name, hierarchy)
        instance_attrs = self._collect_instance_attrs(class_name, hierarchy)
        for attr in instance_attrs:
            all_attrs.add(attr)
        slot_names = self._collect_slot_names(class_name, hierarchy)
        for s in slot_names:
            all_attrs.add(s)

        for attr_name in all_attrs:
            resolution = self.resolve_attribute_access(
                class_name, attr_name, hierarchy
            )
            if resolution.source == "instance_dict":
                result[attr_name] = resolution
            elif (
                resolution.descriptor_info is not None
                and resolution.descriptor_info.has_delete
            ):
                result[attr_name] = resolution
        return result

    # ------------------------------------------------------------------
    # MRO helpers
    # ------------------------------------------------------------------

    def _find_in_mro(
        self,
        class_name: str,
        attr_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Optional[Tuple[str, DescriptorInfo]]:
        """Walk the MRO of *class_name* and return the first class that
        defines *attr_name* together with its :class:`DescriptorInfo`."""
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                # Check builtin knowledge.
                if self._builtin_has_attr(owner, attr_name):
                    info = self._builtin_descriptor_info(owner, attr_name)
                    return (owner, info)
                continue
            if self._class_defines_attr(cls_info, attr_name):
                info = self._do_classify(owner, attr_name, hierarchy)
                return (owner, info)
        return None

    def _has_dunder(
        self,
        class_name: str,
        dunder_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> bool:
        """Return ``True`` if *class_name* (or a base) defines *dunder_name*."""
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                if owner == "object" and dunder_name in {
                    "__init__",
                    "__repr__",
                    "__str__",
                    "__hash__",
                    "__eq__",
                    "__ne__",
                    "__delattr__",
                    "__setattr__",
                    "__getattribute__",
                }:
                    return True
                continue
            if self._class_defines_attr(cls_info, dunder_name):
                return True
        return False

    def _is_data_descriptor(self, info: DescriptorInfo) -> bool:
        """A data descriptor defines both ``__get__`` and ``__set__``."""
        if info.kind in (
            DescriptorKind.DATA,
            DescriptorKind.PROPERTY,
            DescriptorKind.SLOT,
        ):
            return True
        return info.has_get and info.has_set

    def _is_non_data_descriptor(self, info: DescriptorInfo) -> bool:
        """A non-data descriptor defines ``__get__`` but not ``__set__``."""
        if info.kind in (
            DescriptorKind.NON_DATA,
            DescriptorKind.CLASSMETHOD,
            DescriptorKind.STATICMETHOD,
        ):
            return True
        return info.has_get and not info.has_set

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    def _do_classify(
        self,
        class_name: str,
        attr_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> DescriptorInfo:
        """Core classification logic for a single attribute."""
        cls_info = hierarchy.get_class(class_name)

        # 1. Check if it is a __slots__ entry.
        if cls_info is not None and self._class_uses_slots(cls_info):
            slot_names = {
                s for s, _ in self._extract_raw_slots(cls_info)
            }
            if attr_name in slot_names:
                return DescriptorInfo(
                    kind=DescriptorKind.SLOT,
                    defining_class=class_name,
                    has_get=True,
                    has_set=True,
                    has_delete=True,
                    attr_name=attr_name,
                )

        # 2. Check for property / classmethod / staticmethod decorators.
        if cls_info is not None:
            prop = self._find_property_info(cls_info, attr_name)
            if prop is not None:
                has_setter = getattr(prop, "setter_type", None) is not None or getattr(prop, "has_setter", False)
                has_deleter = getattr(prop, "has_deleter", False)
                return DescriptorInfo(
                    kind=DescriptorKind.PROPERTY,
                    defining_class=class_name,
                    has_get=True,
                    has_set=has_setter,
                    has_delete=has_deleter,
                    attr_name=attr_name,
                )

            method = self._find_method(cls_info, attr_name)
            if method is not None:
                decorators = set(getattr(method, "decorators", []))
                if "classmethod" in decorators:
                    return DescriptorInfo(
                        kind=DescriptorKind.CLASSMETHOD,
                        defining_class=class_name,
                        has_get=True,
                        has_set=False,
                        has_delete=False,
                        attr_name=attr_name,
                    )
                if "staticmethod" in decorators:
                    return DescriptorInfo(
                        kind=DescriptorKind.STATICMETHOD,
                        defining_class=class_name,
                        has_get=True,
                        has_set=False,
                        has_delete=False,
                        attr_name=attr_name,
                    )
                # Regular method – functions are non-data descriptors.
                return DescriptorInfo(
                    kind=DescriptorKind.NON_DATA,
                    defining_class=class_name,
                    has_get=True,
                    has_set=False,
                    has_delete=False,
                    attr_name=attr_name,
                )

        # 3. Check the attribute's own type for descriptor dunder methods.
        attr_type = self._get_attr_type(class_name, attr_name, hierarchy)
        if attr_type is not None:
            has_get = self._type_has_dunder(attr_type, "__get__", hierarchy)
            has_set = self._type_has_dunder(attr_type, "__set__", hierarchy)
            has_del = self._type_has_dunder(attr_type, "__delete__", hierarchy)
            if has_get and has_set:
                return DescriptorInfo(
                    kind=DescriptorKind.DATA,
                    defining_class=class_name,
                    has_get=True,
                    has_set=True,
                    has_delete=has_del,
                    attr_name=attr_name,
                )
            if has_get:
                return DescriptorInfo(
                    kind=DescriptorKind.NON_DATA,
                    defining_class=class_name,
                    has_get=True,
                    has_set=False,
                    has_delete=has_del,
                    attr_name=attr_name,
                )

        # 4. Plain attribute – no descriptor protocol involvement.
        return DescriptorInfo(
            kind=DescriptorKind.PLAIN,
            defining_class=class_name,
            has_get=False,
            has_set=False,
            has_delete=False,
            attr_name=attr_name,
        )

    # ------------------------------------------------------------------
    # Helpers – class introspection via ClassHierarchyAnalyzer
    # ------------------------------------------------------------------

    def _iter_mro(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> List[str]:
        """Return the MRO for *class_name* as a list of class names."""
        mro = getattr(hierarchy, "get_mro", None)
        if mro is not None:
            result = mro(class_name)
            if result:
                return list(result)
        # Fallback: walk bases manually in a C3-like BFS.
        return self._compute_mro_fallback(class_name, hierarchy)

    def _compute_mro_fallback(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> List[str]:
        """Simple MRO computation via iterative C3 linearisation fallback."""
        result: List[str] = []
        visited: Set[str] = set()
        queue: List[str] = [class_name]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            cls_info = hierarchy.get_class(current)
            if cls_info is not None:
                bases = getattr(cls_info, "bases", [])
                for b in bases:
                    base_name = b if isinstance(b, str) else getattr(b, "name", str(b))
                    if base_name not in visited:
                        queue.append(base_name)
        if "object" not in visited:
            result.append("object")
        return result

    def _class_defines_attr(
        self,
        cls_info: ClassInfo,
        attr_name: str,
    ) -> bool:
        """Return ``True`` if *cls_info* directly defines *attr_name*."""
        # Check methods.
        methods = getattr(cls_info, "methods", {})
        if isinstance(methods, dict):
            if attr_name in methods:
                return True
        elif isinstance(methods, list):
            if any(getattr(m, "name", None) == attr_name for m in methods):
                return True

        # Check properties.
        properties = getattr(cls_info, "properties", {})
        if isinstance(properties, dict):
            if attr_name in properties:
                return True
        elif isinstance(properties, list):
            if any(getattr(p, "name", None) == attr_name for p in properties):
                return True

        # Check class-level attributes / annotations.
        attrs = getattr(cls_info, "attributes", getattr(cls_info, "attrs", {}))
        if isinstance(attrs, dict):
            if attr_name in attrs:
                return True
        elif isinstance(attrs, list):
            if any(getattr(a, "name", None) == attr_name for a in attrs):
                return True

        # Check __slots__.
        if self._class_uses_slots(cls_info):
            slot_names = {s for s, _ in self._extract_raw_slots(cls_info)}
            if attr_name in slot_names:
                return True

        return False

    def _find_method(
        self,
        cls_info: ClassInfo,
        method_name: str,
    ) -> Optional[MethodInfo]:
        """Locate a :class:`MethodInfo` by name on *cls_info*."""
        methods = getattr(cls_info, "methods", {})
        if isinstance(methods, dict):
            return methods.get(method_name)
        if isinstance(methods, list):
            for m in methods:
                if getattr(m, "name", None) == method_name:
                    return m
        return None

    def _find_property_info(
        self,
        cls_info: ClassInfo,
        prop_name: str,
    ) -> Optional[PropertyInfo]:
        """Locate a :class:`PropertyInfo` by name on *cls_info*."""
        properties = getattr(cls_info, "properties", {})
        if isinstance(properties, dict):
            return properties.get(prop_name)
        if isinstance(properties, list):
            for p in properties:
                if getattr(p, "name", None) == prop_name:
                    return p
        return None

    def _class_uses_slots(self, cls_info: ClassInfo) -> bool:
        """Return ``True`` if the class declares ``__slots__``."""
        slots = getattr(cls_info, "slots", None)
        if slots is not None:
            return True
        attrs = getattr(cls_info, "attributes", getattr(cls_info, "attrs", {}))
        if isinstance(attrs, dict) and "__slots__" in attrs:
            return True
        return False

    def _extract_raw_slots(
        self,
        cls_info: ClassInfo,
    ) -> List[Tuple[str, Optional[str]]]:
        """Return ``[(slot_name, annotation), ...]`` from *cls_info*."""
        slots = getattr(cls_info, "slots", None)
        if slots is None:
            return []
        result: List[Tuple[str, Optional[str]]] = []
        if isinstance(slots, dict):
            for name, ann in slots.items():
                result.append((name, ann if isinstance(ann, str) else None))
        elif isinstance(slots, (list, tuple)):
            for entry in slots:
                if isinstance(entry, str):
                    result.append((entry, None))
                elif isinstance(entry, tuple) and len(entry) >= 2:
                    result.append((str(entry[0]), str(entry[1])))
                else:
                    name = getattr(entry, "name", str(entry))
                    ann = getattr(entry, "annotation", None)
                    result.append((name, ann))
        return result

    def _collect_slot_names(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Set[str]:
        """Collect all slot names across the MRO."""
        names: Set[str] = set()
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            for slot_name, _ in self._extract_raw_slots(cls_info):
                names.add(slot_name)
        return names

    def _slot_annotation(
        self,
        class_name: str,
        slot_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Optional[str]:
        """Return the type annotation for a slot, if available."""
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            for sn, ann in self._extract_raw_slots(cls_info):
                if sn == slot_name:
                    return ann
        return None

    def _slot_has_default(
        self,
        cls_info: ClassInfo,
        slot_name: str,
    ) -> bool:
        """Check if a slot has a default value set as a class variable."""
        attrs = getattr(cls_info, "attributes", getattr(cls_info, "attrs", {}))
        if isinstance(attrs, dict):
            return slot_name in attrs
        return False

    def _collect_instance_attrs(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Dict[str, Optional[str]]:
        """Collect instance attributes assigned in ``__init__`` and friends."""
        result: Dict[str, Optional[str]] = {}
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            instance_attrs = getattr(cls_info, "instance_attributes", None)
            if instance_attrs is None:
                instance_attrs = getattr(cls_info, "instance_attrs", None)
            if instance_attrs is not None:
                if isinstance(instance_attrs, dict):
                    for k, v in instance_attrs.items():
                        if k not in result:
                            result[k] = v if isinstance(v, str) else None
                elif isinstance(instance_attrs, (list, tuple)):
                    for entry in instance_attrs:
                        name = getattr(entry, "name", entry) if not isinstance(entry, str) else entry
                        ann = getattr(entry, "annotation", None) if not isinstance(entry, str) else None
                        if name not in result:
                            result[name] = ann
        return result

    def _collect_class_vars(
        self,
        cls_info: ClassInfo,
    ) -> Set[str]:
        """Return the set of class-level variable names."""
        attrs = getattr(cls_info, "attributes", getattr(cls_info, "attrs", {}))
        if isinstance(attrs, dict):
            return set(attrs.keys())
        if isinstance(attrs, list):
            return {getattr(a, "name", str(a)) for a in attrs}
        return set()

    def _collect_all_mro_attrs(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Set[str]:
        """Gather every attribute name visible in the MRO."""
        names: Set[str] = set()
        for owner in self._iter_mro(class_name, hierarchy):
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            methods = getattr(cls_info, "methods", {})
            if isinstance(methods, dict):
                names.update(methods.keys())
            elif isinstance(methods, list):
                names.update(getattr(m, "name", "") for m in methods)

            properties = getattr(cls_info, "properties", {})
            if isinstance(properties, dict):
                names.update(properties.keys())
            elif isinstance(properties, list):
                names.update(getattr(p, "name", "") for p in properties)

            attrs = getattr(
                cls_info, "attributes", getattr(cls_info, "attrs", {})
            )
            if isinstance(attrs, dict):
                names.update(attrs.keys())
            elif isinstance(attrs, list):
                names.update(getattr(a, "name", "") for a in attrs)

            if self._class_uses_slots(cls_info):
                for s, _ in self._extract_raw_slots(cls_info):
                    names.add(s)

        names.discard("")
        return names

    def _find_local_attr(
        self,
        cls_info: ClassInfo,
        attr_name: str,
    ) -> Optional[DescriptorInfo]:
        """Return a local (non-inherited) DescriptorInfo or None."""
        if not self._class_defines_attr(cls_info, attr_name):
            return None
        name = getattr(cls_info, "name", "")
        return DescriptorInfo(
            kind=DescriptorKind.PLAIN,
            defining_class=name,
            has_get=False,
            has_set=False,
            has_delete=False,
            attr_name=attr_name,
        )

    def _get_attr_type(
        self,
        class_name: str,
        attr_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Optional[str]:
        """Try to determine the type of *attr_name* on *class_name*."""
        cls_info = hierarchy.get_class(class_name)
        if cls_info is None:
            return None
        attrs = getattr(cls_info, "attributes", getattr(cls_info, "attrs", {}))
        if isinstance(attrs, dict):
            val = attrs.get(attr_name)
            if isinstance(val, str):
                return val
            if val is not None:
                return getattr(val, "annotation", getattr(val, "type", None))
        return None

    def _type_has_dunder(
        self,
        type_name: str,
        dunder: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> bool:
        """Check if a type defines a specific dunder (for descriptor protocol)."""
        # Well-known types.
        if type_name == "property":
            return dunder in {"__get__", "__set__", "__delete__"}
        if type_name in ("classmethod", "staticmethod"):
            return dunder == "__get__"
        if type_name == "function":
            return dunder == "__get__"

        cls_info = hierarchy.get_class(type_name)
        if cls_info is None:
            return False
        return self._class_defines_attr(cls_info, dunder)

    def _has_custom_getattribute(
        self,
        class_name: str,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> bool:
        """Return ``True`` if *class_name* overrides ``__getattribute__``."""
        for owner in self._iter_mro(class_name, hierarchy):
            if owner == "object":
                continue
            cls_info = hierarchy.get_class(owner)
            if cls_info is None:
                continue
            if self._class_defines_attr(cls_info, "__getattribute__"):
                return True
        return False

    def _infer_descriptor_value_type(
        self,
        desc_info: DescriptorInfo,
        hierarchy: ClassHierarchyAnalyzer,
    ) -> Optional[str]:
        """Best-effort inference of the value type produced by a descriptor."""
        if desc_info.kind == DescriptorKind.PROPERTY:
            prop_type = self.analyze_property(
                desc_info.defining_class, desc_info.attr_name, hierarchy
            )
            return prop_type.getter_type

        if desc_info.kind == DescriptorKind.CLASSMETHOD:
            cm = self.analyze_classmethod(
                desc_info.defining_class, desc_info.attr_name, hierarchy
            )
            return cm.return_type

        if desc_info.kind == DescriptorKind.STATICMETHOD:
            sm = self.analyze_staticmethod(
                desc_info.defining_class, desc_info.attr_name, hierarchy
            )
            return sm.return_type

        if desc_info.kind == DescriptorKind.SLOT:
            return self._slot_annotation(
                desc_info.defining_class,
                desc_info.attr_name,
                hierarchy,
            )

        if desc_info.kind == DescriptorKind.NON_DATA:
            # Regular methods – try to return the method's return type.
            for owner in [desc_info.defining_class]:
                cls_info = hierarchy.get_class(owner)
                if cls_info is None:
                    continue
                method = self._find_method(cls_info, desc_info.attr_name)
                if method is not None:
                    return getattr(method, "return_type", None)

        if desc_info.kind == DescriptorKind.PLAIN:
            return self._get_attr_type(
                desc_info.defining_class,
                desc_info.attr_name,
                hierarchy,
            )

        return None

    # ------------------------------------------------------------------
    # Builtin knowledge
    # ------------------------------------------------------------------

    def _builtin_has_attr(self, class_name: str, attr_name: str) -> bool:
        """Check builtin knowledge tables."""
        data = self._known_data_descriptors.get(class_name, set())
        non_data = self._known_non_data_descriptors.get(class_name, set())
        return attr_name in data or attr_name in non_data

    def _builtin_descriptor_info(
        self,
        class_name: str,
        attr_name: str,
    ) -> DescriptorInfo:
        """Create a :class:`DescriptorInfo` from builtin knowledge."""
        data = self._known_data_descriptors.get(class_name, set())
        if attr_name in data:
            return DescriptorInfo(
                kind=DescriptorKind.DATA,
                defining_class=class_name,
                has_get=True,
                has_set=True,
                has_delete=False,
                attr_name=attr_name,
            )
        return DescriptorInfo(
            kind=DescriptorKind.NON_DATA,
            defining_class=class_name,
            has_get=True,
            has_set=False,
            has_delete=False,
            attr_name=attr_name,
        )
