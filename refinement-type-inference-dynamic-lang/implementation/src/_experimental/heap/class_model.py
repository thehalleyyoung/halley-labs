"""
Python class system model for refinement type inference.

Models Python's class hierarchy, descriptor protocol, MRO computation
(C3 linearization), attribute resolution, structural subtyping (protocols),
and inheritance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.heap.heap_model import HeapAddress, AbstractValue, AbstractHeap, HeapObject


# ---------------------------------------------------------------------------
# Descriptor modelling
# ---------------------------------------------------------------------------

class DescriptorKind(Enum):
    """Classifies how a descriptor participates in attribute access."""
    DATA_DESCRIPTOR = auto()       # has __get__ and __set__
    NON_DATA_DESCRIPTOR = auto()   # has __get__ only (e.g. plain methods)
    PROPERTY = auto()              # @property
    CLASSMETHOD = auto()           # @classmethod
    STATICMETHOD = auto()          # @staticmethod
    SLOT_DESCRIPTOR = auto()       # __slots__ member descriptor


@dataclass
class DescriptorInfo:
    """Metadata about a single descriptor found on a class."""
    kind: DescriptorKind
    getter_addr: Optional[HeapAddress] = None   # __get__
    setter_addr: Optional[HeapAddress] = None   # __set__
    deleter_addr: Optional[HeapAddress] = None  # __delete__
    underlying_value: Optional[AbstractValue] = None

    # ------------------------------------------------------------------
    @property
    def is_data_descriptor(self) -> bool:
        """A data descriptor defines both __get__ and __set__."""
        return self.kind in (
            DescriptorKind.DATA_DESCRIPTOR,
            DescriptorKind.PROPERTY,
            DescriptorKind.SLOT_DESCRIPTOR,
        )

    @property
    def is_non_data_descriptor(self) -> bool:
        return self.kind in (
            DescriptorKind.NON_DATA_DESCRIPTOR,
            DescriptorKind.CLASSMETHOD,
            DescriptorKind.STATICMETHOD,
        )


# ---------------------------------------------------------------------------
# PythonClass
# ---------------------------------------------------------------------------

@dataclass
class PythonClass:
    """Abstract model of a single Python class."""

    name: str
    address: HeapAddress
    bases: List[HeapAddress] = field(default_factory=list)
    mro: List[HeapAddress] = field(default_factory=list)
    class_attrs: Dict[str, AbstractValue] = field(default_factory=dict)
    instance_attrs: Set[str] = field(default_factory=set)
    descriptors: Dict[str, DescriptorInfo] = field(default_factory=dict)
    slots: Optional[List[str]] = None
    metaclass: Optional[HeapAddress] = None
    is_abstract: bool = False
    abstract_methods: Set[str] = field(default_factory=set)
    is_protocol: bool = False
    protocol_members: Dict[str, AbstractValue] = field(default_factory=dict)
    is_frozen: bool = False
    is_final: bool = False

    # -- query helpers ---------------------------------------------------

    def has_class_attr(self, name: str) -> bool:
        return name in self.class_attrs

    def get_class_attr(self, name: str) -> Optional[AbstractValue]:
        return self.class_attrs.get(name)

    def defines_instance_attr(self, name: str) -> bool:
        return name in self.instance_attrs

    def get_descriptor(self, name: str) -> Optional[DescriptorInfo]:
        return self.descriptors.get(name)

    def is_data_descriptor(self, name: str) -> bool:
        desc = self.descriptors.get(name)
        if desc is None:
            return False
        return desc.is_data_descriptor

    def get_method(self, name: str) -> Optional[HeapAddress]:
        """Return the HeapAddress of *name* if it is a callable class attr."""
        desc = self.descriptors.get(name)
        if desc is not None and desc.getter_addr is not None:
            return desc.getter_addr
        val = self.class_attrs.get(name)
        if val is not None and isinstance(val, AbstractValue):
            if hasattr(val, "address"):
                return getattr(val, "address")
        return None

    def is_subclass_of(
        self,
        other: HeapAddress,
        class_registry: ClassRegistry,
    ) -> bool:
        """True when *other* appears in this class's MRO."""
        if self.address == other:
            return True
        return other in self.mro

    def satisfies_protocol(
        self,
        protocol: PythonClass,
        class_registry: ClassRegistry,
    ) -> bool:
        """Structural subtyping check against a Protocol class."""
        all_methods = self.get_all_methods(class_registry)
        all_attrs = set(self.class_attrs.keys()) | self.instance_attrs
        for member_name in protocol.protocol_members:
            if member_name not in all_methods and member_name not in all_attrs:
                return False
        return True

    def get_all_methods(
        self,
        class_registry: ClassRegistry,
    ) -> Dict[str, HeapAddress]:
        """Collect every method visible through the MRO."""
        result: Dict[str, HeapAddress] = {}
        for cls_addr in reversed(self.mro):
            cls = class_registry.lookup(cls_addr)
            if cls is None:
                continue
            for attr_name, desc in cls.descriptors.items():
                if desc.getter_addr is not None:
                    result[attr_name] = desc.getter_addr
            for attr_name, val in cls.class_attrs.items():
                if attr_name not in result and hasattr(val, "address"):
                    result[attr_name] = getattr(val, "address")
        # Own descriptors / attrs override
        for attr_name, desc in self.descriptors.items():
            if desc.getter_addr is not None:
                result[attr_name] = desc.getter_addr
        for attr_name, val in self.class_attrs.items():
            if attr_name not in result and hasattr(val, "address"):
                result[attr_name] = getattr(val, "address")
        return result

    def join(self, other: PythonClass) -> PythonClass:
        """Lattice join – conservative merge of two class abstractions."""
        if self.address == other.address:
            return self
        merged_class_attrs: Dict[str, AbstractValue] = {}
        for name in self.class_attrs.keys() | other.class_attrs.keys():
            v1 = self.class_attrs.get(name)
            v2 = other.class_attrs.get(name)
            if v1 is not None and v2 is not None:
                if hasattr(v1, "join"):
                    merged_class_attrs[name] = v1.join(v2)  # type: ignore[union-attr]
                else:
                    merged_class_attrs[name] = v1
            elif v1 is not None:
                merged_class_attrs[name] = v1
            else:
                assert v2 is not None
                merged_class_attrs[name] = v2

        merged_instance = self.instance_attrs | other.instance_attrs
        merged_descriptors: Dict[str, DescriptorInfo] = {}
        for name in self.descriptors.keys() | other.descriptors.keys():
            d1 = self.descriptors.get(name)
            d2 = other.descriptors.get(name)
            merged_descriptors[name] = d1 if d1 is not None else d2  # type: ignore[assignment]

        common_bases = [b for b in self.bases if b in other.bases]

        merged_slots: Optional[List[str]] = None
        if self.slots is not None and other.slots is not None:
            merged_slots = list(set(self.slots) | set(other.slots))

        merged_abstract = self.abstract_methods | other.abstract_methods
        merged_protocol_members: Dict[str, AbstractValue] = {}
        for name in self.protocol_members.keys() & other.protocol_members.keys():
            merged_protocol_members[name] = self.protocol_members[name]

        return PythonClass(
            name=f"({self.name} | {other.name})",
            address=self.address,
            bases=common_bases,
            mro=[a for a in self.mro if a in other.mro],
            class_attrs=merged_class_attrs,
            instance_attrs=merged_instance,
            descriptors=merged_descriptors,
            slots=merged_slots,
            metaclass=self.metaclass if self.metaclass == other.metaclass else None,
            is_abstract=self.is_abstract or other.is_abstract,
            abstract_methods=merged_abstract,
            is_protocol=self.is_protocol and other.is_protocol,
            protocol_members=merged_protocol_members,
            is_frozen=self.is_frozen and other.is_frozen,
            is_final=self.is_final and other.is_final,
        )


# ---------------------------------------------------------------------------
# ClassRegistry
# ---------------------------------------------------------------------------

class ClassRegistry:
    """Central registry of all known ``PythonClass`` instances."""

    def __init__(self) -> None:
        self.classes: Dict[HeapAddress, PythonClass] = {}
        self.builtin_classes: Dict[str, HeapAddress] = {}
        self._name_index: Dict[str, HeapAddress] = {}

    # -- mutation ----------------------------------------------------------

    def register(self, cls: PythonClass) -> None:
        self.classes[cls.address] = cls
        self._name_index[cls.name] = cls.address

    def lookup(self, addr: HeapAddress) -> Optional[PythonClass]:
        return self.classes.get(addr)

    def lookup_by_name(self, name: str) -> Optional[PythonClass]:
        addr = self._name_index.get(name)
        if addr is None:
            addr = self.builtin_classes.get(name)
        if addr is None:
            return None
        return self.classes.get(addr)

    def get_builtin(self, name: str) -> HeapAddress:
        addr = self.builtin_classes.get(name)
        if addr is None:
            raise KeyError(f"No builtin class registered for {name!r}")
        return addr

    # -- subclass queries --------------------------------------------------

    def is_subclass(self, sub: HeapAddress, sup: HeapAddress) -> bool:
        if sub == sup:
            return True
        sub_cls = self.lookup(sub)
        if sub_cls is None:
            return False
        return sup in sub_cls.mro

    def common_base(self, addrs: Set[HeapAddress]) -> Optional[HeapAddress]:
        """Find the most-specific class that is a superclass of every *addr*."""
        if not addrs:
            return None
        addr_list = list(addrs)
        first = self.lookup(addr_list[0])
        if first is None:
            return None
        # Candidates are in the MRO of the first class
        candidates = list(first.mro) if first.mro else [first.address]
        if first.address not in candidates:
            candidates.insert(0, first.address)
        for addr in addr_list[1:]:
            cls = self.lookup(addr)
            if cls is None:
                return None
            cls_lineage = set(cls.mro)
            cls_lineage.add(addr)
            candidates = [c for c in candidates if c in cls_lineage]
        if not candidates:
            return self.builtin_classes.get("object")
        return candidates[0]

    # -- builtins ----------------------------------------------------------

    def init_builtins(self) -> None:
        """Populate the registry with the standard builtin types."""
        object_addr = HeapAddress(site="<builtin:object>", context=())
        type_addr = HeapAddress(site="<builtin:type>", context=())

        builtin_specs: List[Tuple[str, List[str]]] = [
            ("object", []),
            ("type", ["object"]),
            ("int", ["object"]),
            ("float", ["object"]),
            ("bool", ["int"]),
            ("str", ["object"]),
            ("bytes", ["object"]),
            ("list", ["object"]),
            ("tuple", ["object"]),
            ("dict", ["object"]),
            ("set", ["object"]),
            ("frozenset", ["object"]),
            ("NoneType", ["object"]),
            ("complex", ["object"]),
            ("range", ["object"]),
            ("slice", ["object"]),
            ("bytearray", ["bytes"]),
            ("memoryview", ["object"]),
        ]

        # First pass – create addresses
        addr_map: Dict[str, HeapAddress] = {}
        for name, _ in builtin_specs:
            addr = HeapAddress(site=f"<builtin:{name}>", context=())
            addr_map[name] = addr
            self.builtin_classes[name] = addr

        # Second pass – create PythonClass objects
        for name, base_names in builtin_specs:
            addr = addr_map[name]
            bases = [addr_map[b] for b in base_names if b in addr_map]
            mro_addrs = [addr] + bases
            # Extend MRO transitively for simple single-inheritance builtins
            for b_name in base_names:
                b_cls = self.classes.get(addr_map.get(b_name, addr))  # type: ignore[arg-type]
                if b_cls is not None:
                    for m in b_cls.mro:
                        if m not in mro_addrs:
                            mro_addrs.append(m)
            cls = PythonClass(
                name=name,
                address=addr,
                bases=bases,
                mro=mro_addrs,
                metaclass=addr_map.get("type"),
            )
            self.register(cls)


# ---------------------------------------------------------------------------
# MRO computation – C3 linearization
# ---------------------------------------------------------------------------

class MROComputer:
    """Compute the Method Resolution Order using C3 linearization."""

    @staticmethod
    def compute_mro(
        cls: PythonClass,
        registry: ClassRegistry,
    ) -> List[HeapAddress]:
        """Main entry point.  Returns the full MRO including *cls* itself."""
        try:
            return MROComputer._c3_linearize(cls, registry)
        except TypeError:
            # Fallback: just the class itself + bases in order
            return [cls.address] + list(cls.bases)

    # ------------------------------------------------------------------

    @staticmethod
    def _c3_linearize(
        cls: PythonClass,
        registry: ClassRegistry,
    ) -> List[HeapAddress]:
        if not cls.bases:
            return [cls.address]

        # Collect linearizations of all bases
        base_mros: List[List[HeapAddress]] = []
        for base_addr in cls.bases:
            base_cls = registry.lookup(base_addr)
            if base_cls is None:
                base_mros.append([base_addr])
            else:
                if base_cls.mro:
                    base_mros.append(list(base_cls.mro))
                else:
                    base_mros.append(
                        MROComputer._c3_linearize(base_cls, registry)
                    )

        # Append the list of direct bases as the final sequence
        sequences = base_mros + [list(cls.bases)]
        result = [cls.address]
        merged = MROComputer._c3_merge(sequences)
        result.extend(merged)
        return result

    @staticmethod
    def _c3_merge(sequences: List[List[HeapAddress]]) -> List[HeapAddress]:
        """Core C3 merge algorithm.

        Raises ``TypeError`` when no consistent MRO exists.
        """
        result: List[HeapAddress] = []
        # Work on copies so we can mutate freely
        seqs = [list(s) for s in sequences]

        while True:
            # Remove empty sequences
            seqs = [s for s in seqs if s]
            if not seqs:
                return result

            # Find a candidate: head of a sequence not in the tail of any other
            candidate: Optional[HeapAddress] = None
            for seq in seqs:
                head = seq[0]
                # Check that head does not appear in any tail
                in_tail = False
                for other_seq in seqs:
                    if head in other_seq[1:]:
                        in_tail = True
                        break
                if not in_tail:
                    candidate = head
                    break

            if candidate is None:
                raise TypeError(
                    "Cannot create a consistent method resolution order (MRO) "
                    "for the given bases"
                )

            result.append(candidate)
            # Remove candidate from all sequences
            for seq in seqs:
                if seq and seq[0] == candidate:
                    seq.pop(0)

    @staticmethod
    def validate_mro(cls: PythonClass, registry: ClassRegistry) -> bool:
        """Return *True* when a valid C3 MRO can be computed."""
        # Check for direct cycles
        visited: Set[HeapAddress] = set()
        stack: List[HeapAddress] = [cls.address]
        while stack:
            cur = stack.pop()
            if cur in visited:
                return False
            visited.add(cur)
            cur_cls = registry.lookup(cur)
            if cur_cls is not None:
                for base in cur_cls.bases:
                    if base == cls.address:
                        return False  # direct cycle
                    stack.append(base)

        try:
            MROComputer._c3_linearize(cls, registry)
            return True
        except TypeError:
            return False


# ---------------------------------------------------------------------------
# Attribute resolution (descriptor protocol)
# ---------------------------------------------------------------------------

class AttributeResolver:
    """Resolve attribute access following Python's descriptor protocol."""

    def __init__(self, heap: AbstractHeap, registry: ClassRegistry) -> None:
        self.heap = heap
        self.registry = registry

    # -- getattr -----------------------------------------------------------

    def resolve_getattr(
        self,
        obj_addr: HeapAddress,
        name: str,
    ) -> Tuple[Optional[AbstractValue], str]:
        """Resolve ``obj.name``.

        Returns ``(value, source)`` where *source* is one of
        ``'data_descriptor'``, ``'instance'``, ``'class'``,
        ``'non_data_descriptor'``, ``'getattr'``, or ``'not_found'``.
        """
        obj = self.heap.lookup(obj_addr) if hasattr(self.heap, "lookup") else None
        class_addr = self._get_class(obj_addr, obj)
        if class_addr is None:
            return (None, "not_found")

        cls = self.registry.lookup(class_addr)
        if cls is None:
            return (None, "not_found")

        mro = cls.mro if cls.mro else [class_addr]

        # Step 1 – data descriptor in MRO
        for mro_addr in mro:
            mro_cls = self.registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            desc = mro_cls.get_descriptor(name)
            if desc is not None and desc.is_data_descriptor:
                val = self.invoke_descriptor_get(desc, obj_addr, class_addr)
                return (val, "data_descriptor")

        # Step 2 – instance __dict__
        if obj is not None and hasattr(obj, "fields"):
            inst_val = obj.fields.get(name) if isinstance(obj.fields, dict) else None  # type: ignore[union-attr]
            if inst_val is not None:
                return (inst_val, "instance")
        if obj is not None and hasattr(obj, "attrs"):
            inst_val = obj.attrs.get(name) if isinstance(obj.attrs, dict) else None  # type: ignore[union-attr]
            if inst_val is not None:
                return (inst_val, "instance")

        # Step 3 – non-data descriptors & plain class attributes in MRO
        for mro_addr in mro:
            mro_cls = self.registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            desc = mro_cls.get_descriptor(name)
            if desc is not None and desc.is_non_data_descriptor:
                val = self.invoke_descriptor_get(desc, obj_addr, class_addr)
                return (val, "non_data_descriptor")
            class_val = mro_cls.get_class_attr(name)
            if class_val is not None:
                return (class_val, "class")

        # Step 4 – __getattr__ fallback
        for mro_addr in mro:
            mro_cls = self.registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            if mro_cls.has_class_attr("__getattr__"):
                getattr_val = mro_cls.get_class_attr("__getattr__")
                return (getattr_val, "getattr")

        return (None, "not_found")

    # -- setattr -----------------------------------------------------------

    def resolve_setattr(
        self,
        obj_addr: HeapAddress,
        name: str,
        val: AbstractValue,
    ) -> AbstractHeap:
        """Resolve ``obj.name = val``."""
        obj = self.heap.lookup(obj_addr) if hasattr(self.heap, "lookup") else None
        class_addr = self._get_class(obj_addr, obj)

        # Step 1 – data descriptor with __set__
        if class_addr is not None:
            cls = self.registry.lookup(class_addr)
            if cls is not None:
                mro = cls.mro if cls.mro else [class_addr]
                for mro_addr in mro:
                    mro_cls = self.registry.lookup(mro_addr)
                    if mro_cls is None:
                        continue
                    desc = mro_cls.get_descriptor(name)
                    if desc is not None and desc.is_data_descriptor and desc.setter_addr is not None:
                        return self.invoke_descriptor_set(desc, obj_addr, val)

        # Step 2 – __slots__ check
        if class_addr is not None:
            cls = self.registry.lookup(class_addr)
            if cls is not None and cls.slots is not None:
                if name not in cls.slots and name != "__dict__":
                    # Cannot set attribute – slots restriction
                    return self.heap

        # Step 3 – set in instance __dict__
        return self._set_instance_attr(obj_addr, name, val)

    # -- delattr -----------------------------------------------------------

    def resolve_delattr(
        self,
        obj_addr: HeapAddress,
        name: str,
    ) -> AbstractHeap:
        """Resolve ``del obj.name``."""
        obj = self.heap.lookup(obj_addr) if hasattr(self.heap, "lookup") else None
        class_addr = self._get_class(obj_addr, obj)

        if class_addr is not None:
            cls = self.registry.lookup(class_addr)
            if cls is not None:
                mro = cls.mro if cls.mro else [class_addr]
                for mro_addr in mro:
                    mro_cls = self.registry.lookup(mro_addr)
                    if mro_cls is None:
                        continue
                    desc = mro_cls.get_descriptor(name)
                    if desc is not None and desc.is_data_descriptor and desc.deleter_addr is not None:
                        # Invoke __delete__
                        return self.heap

        # Remove from instance dict
        return self._del_instance_attr(obj_addr, name)

    # -- class-level access ------------------------------------------------

    def resolve_class_getattr(
        self,
        class_addr: HeapAddress,
        name: str,
    ) -> Optional[AbstractValue]:
        """Resolve ``ClassName.attr``."""
        cls = self.registry.lookup(class_addr)
        if cls is None:
            return None

        mro = cls.mro if cls.mro else [class_addr]
        for mro_addr in mro:
            mro_cls = self.registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            desc = mro_cls.get_descriptor(name)
            if desc is not None:
                if desc.kind == DescriptorKind.STATICMETHOD:
                    return desc.underlying_value
                if desc.kind == DescriptorKind.CLASSMETHOD:
                    return desc.underlying_value
                if desc.getter_addr is not None:
                    return desc.underlying_value
            val = mro_cls.get_class_attr(name)
            if val is not None:
                return val
        return None

    # -- super() resolution ------------------------------------------------

    def resolve_super(
        self,
        obj_addr: HeapAddress,
        class_addr: HeapAddress,
        name: str,
    ) -> Optional[AbstractValue]:
        """Resolve ``super().name`` where *class_addr* is the class calling super."""
        cls = self.registry.lookup(class_addr)
        if cls is None:
            return None

        mro = cls.mro if cls.mro else [class_addr]
        # Find class_addr in MRO, start searching after it
        try:
            idx = mro.index(class_addr)
        except ValueError:
            idx = -1
        search_mro = mro[idx + 1:] if idx >= 0 else mro

        for mro_addr in search_mro:
            mro_cls = self.registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            desc = mro_cls.get_descriptor(name)
            if desc is not None:
                return self.invoke_descriptor_get(desc, obj_addr, class_addr)
            val = mro_cls.get_class_attr(name)
            if val is not None:
                return val
        return None

    # -- descriptor invocation helpers -------------------------------------

    def invoke_descriptor_get(
        self,
        desc_info: DescriptorInfo,
        obj_addr: HeapAddress,
        class_addr: HeapAddress,
    ) -> Optional[AbstractValue]:
        """Invoke ``descriptor.__get__(obj, type)`` abstractly."""
        if desc_info.kind == DescriptorKind.STATICMETHOD:
            return desc_info.underlying_value

        if desc_info.kind == DescriptorKind.CLASSMETHOD:
            return desc_info.underlying_value

        if desc_info.kind == DescriptorKind.PROPERTY:
            # The getter would produce a fresh value; return underlying as
            # a best-effort approximation.
            return desc_info.underlying_value

        if desc_info.kind == DescriptorKind.SLOT_DESCRIPTOR:
            obj = self.heap.lookup(obj_addr) if hasattr(self.heap, "lookup") else None
            if obj is not None and hasattr(obj, "fields") and isinstance(obj.fields, dict):  # type: ignore[union-attr]
                return obj.fields.get(desc_info.underlying_value)  # type: ignore[union-attr]
            return desc_info.underlying_value

        # General descriptor (__get__)
        return desc_info.underlying_value

    def invoke_descriptor_set(
        self,
        desc_info: DescriptorInfo,
        obj_addr: HeapAddress,
        val: AbstractValue,
    ) -> AbstractHeap:
        """Invoke ``descriptor.__set__(obj, val)`` abstractly."""
        if desc_info.kind == DescriptorKind.SLOT_DESCRIPTOR:
            return self._set_instance_attr(obj_addr, str(desc_info.underlying_value), val)

        if desc_info.kind == DescriptorKind.PROPERTY:
            # Property setter – side effects modeled by updating heap
            return self.heap

        # General data descriptor __set__
        return self.heap

    # -- private helpers ---------------------------------------------------

    def _get_class(
        self,
        obj_addr: HeapAddress,
        obj: Optional[HeapObject],
    ) -> Optional[HeapAddress]:
        """Determine the class (type) of an object."""
        if obj is not None:
            if hasattr(obj, "class_ref") and obj.class_ref is not None:  # type: ignore[union-attr]
                return obj.class_ref  # type: ignore[return-value]
            if hasattr(obj, "type_addr") and obj.type_addr is not None:  # type: ignore[union-attr]
                return obj.type_addr  # type: ignore[return-value]
        # Fallback: try to match address site to a builtin
        site = obj_addr.site
        for builtin_name, builtin_addr in self.registry.builtin_classes.items():
            if builtin_name in site:
                return builtin_addr
        return None

    def _set_instance_attr(
        self,
        obj_addr: HeapAddress,
        name: str,
        val: AbstractValue,
    ) -> AbstractHeap:
        """Set an attribute in the instance's __dict__ on the abstract heap."""
        if hasattr(self.heap, "write_field"):
            return self.heap.write_field(obj_addr, name, val)  # type: ignore[return-value]
        if hasattr(self.heap, "store"):
            return self.heap.store(obj_addr, name, val)  # type: ignore[return-value]
        return self.heap

    def _del_instance_attr(
        self,
        obj_addr: HeapAddress,
        name: str,
    ) -> AbstractHeap:
        """Delete an attribute from the instance's __dict__."""
        if hasattr(self.heap, "delete_field"):
            return self.heap.delete_field(obj_addr, name)  # type: ignore[return-value]
        return self.heap


# ---------------------------------------------------------------------------
# Protocol checking (structural subtyping)
# ---------------------------------------------------------------------------

@dataclass
class ProtocolCheckResult:
    """Outcome of checking a class against a Protocol."""
    compliant: bool
    missing_members: Set[str] = field(default_factory=set)
    incompatible_members: Dict[str, str] = field(default_factory=dict)
    extra_members: Set[str] = field(default_factory=set)


class ProtocolChecker:
    """Check structural subtyping compliance against ``typing.Protocol``."""

    @staticmethod
    def check_protocol(
        obj_type: PythonClass,
        protocol: PythonClass,
        registry: ClassRegistry,
    ) -> ProtocolCheckResult:
        """Full protocol compliance check."""
        missing = ProtocolChecker.get_missing_members(obj_type, protocol)
        incompatible = ProtocolChecker.get_incompatible_members(
            obj_type, protocol
        )

        obj_names = (
            set(obj_type.class_attrs.keys())
            | obj_type.instance_attrs
            | set(obj_type.descriptors.keys())
        )
        protocol_names = set(protocol.protocol_members.keys())
        extra = obj_names - protocol_names

        compliant = len(missing) == 0 and len(incompatible) == 0

        return ProtocolCheckResult(
            compliant=compliant,
            missing_members=missing,
            incompatible_members=incompatible,
            extra_members=extra,
        )

    @staticmethod
    def check_method_signature(
        impl_method: Optional[AbstractValue],
        protocol_method: AbstractValue,
    ) -> bool:
        """Compare an implementation method's signature against the protocol.

        In the abstract domain we may not have full signature information, so
        we perform a conservative check: if both values carry arity or type
        info we compare them, otherwise we assume compatibility.
        """
        if impl_method is None:
            return False

        # If the abstract values expose an arity attribute, compare them.
        impl_arity = getattr(impl_method, "arity", None)
        proto_arity = getattr(protocol_method, "arity", None)
        if impl_arity is not None and proto_arity is not None:
            if impl_arity < proto_arity:
                return False

        # If the values expose a return type attribute, check compatibility.
        impl_ret = getattr(impl_method, "return_type", None)
        proto_ret = getattr(protocol_method, "return_type", None)
        if impl_ret is not None and proto_ret is not None:
            if hasattr(proto_ret, "is_subtype"):
                if not impl_ret.is_subtype(proto_ret):  # type: ignore[union-attr]
                    return False

        return True

    @staticmethod
    def get_missing_members(
        obj_type: PythonClass,
        protocol: PythonClass,
    ) -> Set[str]:
        """Return protocol members not present on *obj_type*."""
        available = (
            set(obj_type.class_attrs.keys())
            | obj_type.instance_attrs
            | set(obj_type.descriptors.keys())
        )
        required = set(protocol.protocol_members.keys())
        return required - available

    @staticmethod
    def get_incompatible_members(
        obj_type: PythonClass,
        protocol: PythonClass,
    ) -> Dict[str, str]:
        """Return members present but with incompatible types/signatures."""
        result: Dict[str, str] = {}
        available = set(obj_type.class_attrs.keys()) | set(obj_type.descriptors.keys())

        for member_name, proto_val in protocol.protocol_members.items():
            if member_name not in available:
                continue  # missing, handled elsewhere
            impl_val = obj_type.class_attrs.get(member_name)
            if impl_val is None:
                desc = obj_type.descriptors.get(member_name)
                impl_val = desc.underlying_value if desc is not None else None

            if not ProtocolChecker.check_method_signature(impl_val, proto_val):
                result[member_name] = (
                    f"Signature of '{member_name}' is incompatible with protocol"
                )
        return result


# ---------------------------------------------------------------------------
# ClassBuilder – construct PythonClass from various sources
# ---------------------------------------------------------------------------

class ClassBuilder:
    """Build ``PythonClass`` instances from AST analysis or runtime info."""

    @staticmethod
    def build_from_dict(
        name: str,
        bases: List[HeapAddress],
        namespace: Dict[str, AbstractValue],
        metaclass: Optional[HeapAddress],
        registry: ClassRegistry,
    ) -> PythonClass:
        """Construct a ``PythonClass`` from a class body namespace dict."""
        addr = HeapAddress(site=f"<class:{name}>", context=())

        descriptors = ClassBuilder.detect_descriptors(namespace)
        init_attrs = ClassBuilder.detect_init_attrs(
            namespace.get("__init__")
        )
        abstract_methods = ClassBuilder.detect_abstract_methods(namespace)
        slots = ClassBuilder.detect_slots(namespace)

        # Filter descriptors out of plain class_attrs
        class_attrs: Dict[str, AbstractValue] = {}
        for k, v in namespace.items():
            if k.startswith("__") and k.endswith("__") and k in (
                "__module__", "__qualname__", "__doc__",
            ):
                continue
            if k not in descriptors:
                class_attrs[k] = v

        is_abstract = len(abstract_methods) > 0 or _has_abc_meta(metaclass, registry)
        is_protocol = _check_protocol_base(bases, registry)

        protocol_members: Dict[str, AbstractValue] = {}
        if is_protocol:
            for k, v in class_attrs.items():
                if not k.startswith("_"):
                    protocol_members[k] = v
            for k, d in descriptors.items():
                if not k.startswith("_") and d.underlying_value is not None:
                    protocol_members[k] = d.underlying_value

        cls = PythonClass(
            name=name,
            address=addr,
            bases=bases,
            class_attrs=class_attrs,
            instance_attrs=init_attrs,
            descriptors=descriptors,
            slots=slots,
            metaclass=metaclass,
            is_abstract=is_abstract,
            abstract_methods=abstract_methods,
            is_protocol=is_protocol,
            protocol_members=protocol_members,
            is_frozen=_check_frozen(namespace),
            is_final=_check_final(namespace),
        )

        # Compute MRO
        registry.register(cls)
        cls.mro = MROComputer.compute_mro(cls, registry)

        return cls

    @staticmethod
    def build_builtin(
        name: str,
        methods: Dict[str, HeapAddress],
        bases: List[HeapAddress],
    ) -> PythonClass:
        """Construct a ``PythonClass`` for a builtin type."""
        addr = HeapAddress(site=f"<builtin:{name}>", context=())
        descriptors: Dict[str, DescriptorInfo] = {}
        for method_name, method_addr in methods.items():
            descriptors[method_name] = DescriptorInfo(
                kind=DescriptorKind.NON_DATA_DESCRIPTOR,
                getter_addr=method_addr,
            )
        return PythonClass(
            name=name,
            address=addr,
            bases=bases,
            mro=[addr] + bases,
            descriptors=descriptors,
        )

    @staticmethod
    def detect_descriptors(
        namespace: Dict[str, AbstractValue],
    ) -> Dict[str, DescriptorInfo]:
        """Scan a namespace for descriptor-like objects."""
        result: Dict[str, DescriptorInfo] = {}
        for attr_name, val in namespace.items():
            kind = _classify_descriptor(val)
            if kind is None:
                continue
            getter = getattr(val, "fget", None) or getattr(val, "__get__", None)
            setter = getattr(val, "fset", None) or getattr(val, "__set__", None)
            deleter = getattr(val, "fdel", None) or getattr(val, "__delete__", None)

            getter_addr = _addr_of(getter)
            setter_addr = _addr_of(setter)
            deleter_addr = _addr_of(deleter)

            result[attr_name] = DescriptorInfo(
                kind=kind,
                getter_addr=getter_addr,
                setter_addr=setter_addr,
                deleter_addr=deleter_addr,
                underlying_value=val,
            )
        return result

    @staticmethod
    def detect_init_attrs(init_body: Optional[AbstractValue]) -> Set[str]:
        """Detect attributes assigned in ``__init__``.

        If *init_body* carries an ``assigned_attrs`` or ``body_attrs``
        attribute (set by a prior AST pass), use it.  Otherwise return
        an empty set.
        """
        if init_body is None:
            return set()
        attrs = getattr(init_body, "assigned_attrs", None)
        if attrs is not None and isinstance(attrs, (set, frozenset)):
            return set(attrs)
        attrs = getattr(init_body, "body_attrs", None)
        if attrs is not None and isinstance(attrs, (set, frozenset)):
            return set(attrs)
        # Try to extract self.<name> assignments from an AST-like structure
        stmts = getattr(init_body, "statements", None)
        if stmts is not None:
            found: Set[str] = set()
            for stmt in stmts:
                target = getattr(stmt, "target", None)
                if target is not None:
                    attr = getattr(target, "attr", None)
                    recv = getattr(target, "value", None)
                    if (
                        attr is not None
                        and recv is not None
                        and getattr(recv, "id", None) == "self"
                    ):
                        found.add(attr)
            return found
        return set()

    @staticmethod
    def detect_abstract_methods(
        namespace: Dict[str, AbstractValue],
    ) -> Set[str]:
        """Return names of methods decorated with ``@abstractmethod``."""
        abstract: Set[str] = set()
        for attr_name, val in namespace.items():
            if getattr(val, "is_abstract", False):
                abstract.add(attr_name)
            if getattr(val, "__isabstractmethod__", False):
                abstract.add(attr_name)
        return abstract

    @staticmethod
    def detect_slots(
        namespace: Dict[str, AbstractValue],
    ) -> Optional[List[str]]:
        """Return ``__slots__`` if present in the namespace."""
        slots_val = namespace.get("__slots__")
        if slots_val is None:
            return None
        if isinstance(slots_val, (list, tuple)):
            return [str(s) for s in slots_val]
        if hasattr(slots_val, "elements"):
            return [str(e) for e in slots_val.elements]  # type: ignore[union-attr]
        raw = getattr(slots_val, "value", None)
        if isinstance(raw, (list, tuple)):
            return [str(s) for s in raw]
        return None


# ---------------------------------------------------------------------------
# InheritanceAnalyzer
# ---------------------------------------------------------------------------

class InheritanceAnalyzer:
    """Analyse inheritance hierarchies for common issues."""

    @staticmethod
    def find_diamond(
        cls: PythonClass,
        registry: ClassRegistry,
    ) -> Optional[List[List[HeapAddress]]]:
        """Return diamond inheritance paths, if any.

        Each inner list is a path from *cls* to a shared ancestor that is
        reached through more than one route.
        """
        if len(cls.bases) < 2:
            return None

        # Collect ancestor sets for each base
        ancestor_sets: Dict[HeapAddress, Set[HeapAddress]] = {}
        for base_addr in cls.bases:
            ancestors: Set[HeapAddress] = set()
            _collect_ancestors(base_addr, registry, ancestors)
            ancestor_sets[base_addr] = ancestors

        # Find shared ancestors reachable via multiple bases
        shared: Set[HeapAddress] = set()
        base_list = list(cls.bases)
        for i in range(len(base_list)):
            for j in range(i + 1, len(base_list)):
                common = ancestor_sets[base_list[i]] & ancestor_sets[base_list[j]]
                shared |= common

        if not shared:
            return None

        # Build paths to each shared ancestor
        paths: List[List[HeapAddress]] = []
        for ancestor in shared:
            for base_addr in cls.bases:
                path = _find_path(cls.address, ancestor, registry)
                if path is not None and len(path) > 1:
                    paths.append(path)

        return paths if paths else None

    @staticmethod
    def find_mro_conflicts(
        cls: PythonClass,
        registry: ClassRegistry,
    ) -> List[str]:
        """Find methods with conflicting definitions in the MRO.

        A conflict exists when two different bases (that are not in a direct
        sub/super relationship) both define the same method.
        """
        conflicts: List[str] = []
        method_sources: Dict[str, List[HeapAddress]] = {}

        mro = cls.mro if cls.mro else [cls.address]
        for mro_addr in mro:
            if mro_addr == cls.address:
                continue
            mro_cls = registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            for attr_name in mro_cls.class_attrs:
                method_sources.setdefault(attr_name, []).append(mro_addr)
            for attr_name in mro_cls.descriptors:
                if attr_name not in method_sources:
                    method_sources.setdefault(attr_name, []).append(mro_addr)

        for method_name, sources in method_sources.items():
            if len(sources) < 2:
                continue
            # Check if sources are in a sub/super relationship
            conflicting = False
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    if not registry.is_subclass(sources[i], sources[j]) and \
                       not registry.is_subclass(sources[j], sources[i]):
                        conflicting = True
                        break
                if conflicting:
                    break
            if conflicting:
                source_names = []
                for s in sources:
                    s_cls = registry.lookup(s)
                    source_names.append(s_cls.name if s_cls else str(s))
                conflicts.append(
                    f"'{method_name}' has conflicting definitions in "
                    f"{', '.join(source_names)}"
                )

        return conflicts

    @staticmethod
    def find_override_errors(
        cls: PythonClass,
        registry: ClassRegistry,
    ) -> List[str]:
        """Detect Liskov Substitution Principle violations.

        Checks that overriding methods have compatible signatures (where
        signature info is available from the abstract domain).
        """
        errors: List[str] = []
        mro = cls.mro if cls.mro else [cls.address]

        for method_name in cls.class_attrs:
            child_val = cls.class_attrs[method_name]
            # Walk MRO (skip self) to find the overridden version
            for mro_addr in mro:
                if mro_addr == cls.address:
                    continue
                parent = registry.lookup(mro_addr)
                if parent is None:
                    continue
                parent_val = parent.class_attrs.get(method_name)
                if parent_val is None:
                    continue

                # Compare arities
                child_arity = getattr(child_val, "arity", None)
                parent_arity = getattr(parent_val, "arity", None)
                if child_arity is not None and parent_arity is not None:
                    if child_arity < parent_arity:
                        parent_name = parent.name
                        errors.append(
                            f"'{method_name}' in '{cls.name}' has fewer "
                            f"parameters ({child_arity}) than in "
                            f"'{parent_name}' ({parent_arity})"
                        )

                # Compare return types if available
                child_ret = getattr(child_val, "return_type", None)
                parent_ret = getattr(parent_val, "return_type", None)
                if child_ret is not None and parent_ret is not None:
                    if hasattr(child_ret, "is_subtype"):
                        if not child_ret.is_subtype(parent_ret):
                            errors.append(
                                f"'{method_name}' in '{cls.name}' has return "
                                f"type incompatible with '{parent.name}'"
                            )

                # Check if parent method is final
                if getattr(parent_val, "is_final", False):
                    errors.append(
                        f"'{method_name}' in '{cls.name}' overrides a "
                        f"@final method from '{parent.name}'"
                    )
                break  # only check immediate override

        return errors

    @staticmethod
    def find_abstract_violations(
        cls: PythonClass,
        registry: ClassRegistry,
    ) -> List[str]:
        """Find abstract methods that are not implemented by *cls*.

        Walks the MRO to collect all abstract methods, then checks that
        *cls* or one of its bases provides a concrete implementation.
        """
        if cls.is_abstract:
            return []  # abstract classes don't need to implement everything

        all_abstract: Set[str] = set()
        mro = cls.mro if cls.mro else [cls.address]

        for mro_addr in mro:
            mro_cls = registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            all_abstract |= mro_cls.abstract_methods

        # Now remove methods that have a concrete implementation
        concrete: Set[str] = set()
        for mro_addr in mro:
            mro_cls = registry.lookup(mro_addr)
            if mro_cls is None:
                continue
            for attr_name in mro_cls.class_attrs:
                val = mro_cls.class_attrs[attr_name]
                if not getattr(val, "is_abstract", False) and \
                   not getattr(val, "__isabstractmethod__", False):
                    concrete.add(attr_name)
            for attr_name, desc in mro_cls.descriptors.items():
                if desc.getter_addr is not None:
                    uv = desc.underlying_value
                    if uv is not None and not getattr(uv, "is_abstract", False):
                        concrete.add(attr_name)

        unimplemented = all_abstract - concrete
        violations: List[str] = []
        for method_name in sorted(unimplemented):
            # Find which base declared it abstract
            source = "unknown"
            for mro_addr in mro:
                mro_cls = registry.lookup(mro_addr)
                if mro_cls is not None and method_name in mro_cls.abstract_methods:
                    source = mro_cls.name
                    break
            violations.append(
                f"'{cls.name}' does not implement abstract method "
                f"'{method_name}' from '{source}'"
            )

        return violations


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _collect_ancestors(
    addr: HeapAddress,
    registry: ClassRegistry,
    out: Set[HeapAddress],
) -> None:
    """Recursively collect all ancestors of *addr* into *out*."""
    out.add(addr)
    cls = registry.lookup(addr)
    if cls is None:
        return
    for base in cls.bases:
        if base not in out:
            _collect_ancestors(base, registry, out)


def _find_path(
    start: HeapAddress,
    target: HeapAddress,
    registry: ClassRegistry,
) -> Optional[List[HeapAddress]]:
    """BFS to find a path from *start* to *target* through inheritance."""
    from collections import deque

    queue: deque[List[HeapAddress]] = deque([[start]])
    visited: Set[HeapAddress] = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == target:
            return path
        cls = registry.lookup(current)
        if cls is None:
            continue
        for base in cls.bases:
            if base not in visited:
                visited.add(base)
                queue.append(path + [base])
    return None


def _classify_descriptor(val: AbstractValue) -> Optional[DescriptorKind]:
    """Attempt to classify an abstract value as a descriptor."""
    type_name = type(val).__name__.lower()

    if "property" in type_name:
        return DescriptorKind.PROPERTY
    if "classmethod" in type_name:
        return DescriptorKind.CLASSMETHOD
    if "staticmethod" in type_name:
        return DescriptorKind.STATICMETHOD

    # Check for descriptor protocol attributes
    has_get = hasattr(val, "__get__") or getattr(val, "has_get", False)
    has_set = hasattr(val, "__set__") or getattr(val, "has_set", False)

    # Check tag-based classification from the abstract domain
    tag = getattr(val, "descriptor_kind", None)
    if tag is not None:
        mapping = {
            "property": DescriptorKind.PROPERTY,
            "classmethod": DescriptorKind.CLASSMETHOD,
            "staticmethod": DescriptorKind.STATICMETHOD,
            "data": DescriptorKind.DATA_DESCRIPTOR,
            "non_data": DescriptorKind.NON_DATA_DESCRIPTOR,
            "slot": DescriptorKind.SLOT_DESCRIPTOR,
        }
        return mapping.get(str(tag))

    if has_get and has_set:
        return DescriptorKind.DATA_DESCRIPTOR
    if has_get:
        return DescriptorKind.NON_DATA_DESCRIPTOR

    return None


def _addr_of(obj: object) -> Optional[HeapAddress]:
    """Extract a ``HeapAddress`` from an object if possible."""
    if obj is None:
        return None
    if isinstance(obj, HeapAddress):
        return obj
    addr = getattr(obj, "address", None)
    if isinstance(addr, HeapAddress):
        return addr
    addr = getattr(obj, "heap_address", None)
    if isinstance(addr, HeapAddress):
        return addr
    return None


def _has_abc_meta(
    metaclass: Optional[HeapAddress],
    registry: ClassRegistry,
) -> bool:
    """Check if the metaclass is ABCMeta."""
    if metaclass is None:
        return False
    meta_cls = registry.lookup(metaclass)
    if meta_cls is None:
        return False
    return "ABCMeta" in meta_cls.name or "abc" in meta_cls.name.lower()


def _check_protocol_base(
    bases: List[HeapAddress],
    registry: ClassRegistry,
) -> bool:
    """Check if any base is ``typing.Protocol``."""
    for base_addr in bases:
        cls = registry.lookup(base_addr)
        if cls is None:
            continue
        if cls.name == "Protocol" or cls.is_protocol:
            return True
    return False


def _check_frozen(namespace: Dict[str, AbstractValue]) -> bool:
    """Heuristic: check if the class is a frozen dataclass or NamedTuple."""
    frozen = namespace.get("__dataclass_params__")
    if frozen is not None:
        f = getattr(frozen, "frozen", None)
        if f:
            return True
    # Check for a _frozen flag sometimes set by codegen
    if namespace.get("_frozen"):
        return True
    # Check NamedTuple by looking for _fields
    if "_fields" in namespace and "_asdict" in namespace:
        return True
    return False


def _check_final(namespace: Dict[str, AbstractValue]) -> bool:
    """Check for ``@final`` decorator marker."""
    final_marker = namespace.get("__final__")
    if final_marker is not None:
        return bool(final_marker)
    return False
