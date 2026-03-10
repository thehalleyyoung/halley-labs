"""
usability_oracle.aria.types — Data types for WAI-ARIA taxonomy and conformance.

Provides immutable value types for representing ARIA roles, properties,
states, role relationships, and conformance checking results per the
WAI-ARIA 1.2 specification (W3C Recommendation, 6 June 2023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class RoleCategory(Enum):
    """WAI-ARIA 1.2 role categories (§5.3)."""

    ABSTRACT = "abstract"
    """Abstract roles used only for ontological purposes (never exposed)."""

    WIDGET = "widget"
    """Interactive UI widgets (button, checkbox, slider, …)."""

    DOCUMENT_STRUCTURE = "document_structure"
    """Structural elements (article, heading, list, …)."""

    LANDMARK = "landmark"
    """Navigational landmark regions (banner, main, navigation, …)."""

    LIVE_REGION = "live_region"
    """Regions with dynamically changing content (alert, log, status, …)."""

    WINDOW = "window"
    """Window-like constructs (alertdialog, dialog)."""


@unique
class PropertyType(Enum):
    """Data type of an ARIA property or state value."""

    BOOLEAN = "boolean"
    ID_REFERENCE = "id_reference"
    ID_REFERENCE_LIST = "id_reference_list"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    TOKEN = "token"
    TOKEN_LIST = "token_list"
    TRISTATE = "tristate"


@unique
class ConformanceLevel(Enum):
    """ARIA conformance level for a role usage."""

    CONFORMING = "conforming"
    """Fully conforming to the specification."""

    VIOLATION = "violation"
    """One or more required properties or states are missing/invalid."""

    WARNING = "warning"
    """Deprecated usage or suboptimal pattern."""

    NOT_APPLICABLE = "not_applicable"
    """Role is abstract and must not appear in content."""


# ═══════════════════════════════════════════════════════════════════════════
# AriaProperty
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AriaProperty:
    """An ARIA property definition (aria-* attribute).

    Properties are characteristics that are essential to the nature of a
    widget, and whose values are unlikely to change over the lifecycle.

    Attributes:
        name: Property name without the ``aria-`` prefix
            (e.g. ``"label"``, ``"describedby"``).
        property_type: Data type of the property value.
        default_value: Default value if not explicitly specified.
        is_global: ``True`` if this property applies to all roles.
        deprecated: ``True`` if deprecated in ARIA 1.2.
        description: Human-readable description from the spec.
    """

    name: str
    property_type: PropertyType
    default_value: Optional[str] = None
    is_global: bool = False
    deprecated: bool = False
    description: str = ""

    @property
    def full_name(self) -> str:
        """Full attribute name with ``aria-`` prefix."""
        return f"aria-{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "property_type": self.property_type.value,
            "default_value": self.default_value,
            "is_global": self.is_global,
            "deprecated": self.deprecated,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AriaProperty:
        return cls(
            name=str(d["name"]),
            property_type=PropertyType(d["property_type"]),
            default_value=d.get("default_value"),
            is_global=bool(d.get("is_global", False)),
            deprecated=bool(d.get("deprecated", False)),
            description=str(d.get("description", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# AriaState
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AriaState:
    """An ARIA state definition.

    States are dynamic properties whose values change in response to
    user interaction (e.g. ``aria-checked``, ``aria-expanded``).

    Attributes:
        name: State name without the ``aria-`` prefix.
        state_type: Data type of the state value.
        default_value: Default value when not set.
        is_global: Whether this state is global (applies to all roles).
        description: Human-readable description.
    """

    name: str
    state_type: PropertyType
    default_value: Optional[str] = None
    is_global: bool = False
    description: str = ""

    @property
    def full_name(self) -> str:
        """Full attribute name with ``aria-`` prefix."""
        return f"aria-{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state_type": self.state_type.value,
            "default_value": self.default_value,
            "is_global": self.is_global,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AriaState:
        return cls(
            name=str(d["name"]),
            state_type=PropertyType(d["state_type"]),
            default_value=d.get("default_value"),
            is_global=bool(d.get("is_global", False)),
            description=str(d.get("description", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# AriaRole
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AriaRole:
    """Complete ARIA role descriptor.

    Encodes a single WAI-ARIA 1.2 role with its associated metadata:
    superclass chain, allowed children/parents, required and supported
    properties and states.

    Attributes:
        name: Role name (e.g. ``"button"``, ``"navigation"``).
        category: Ontological category.
        superclass_roles: Immediate superclass role names in the taxonomy.
        subclass_roles: Immediate subclass role names.
        required_properties: Properties that *must* be set on this role.
        supported_properties: Properties *allowed* (but not required).
        required_states: States that *must* be set.
        supported_states: States *allowed* (but not required).
        required_owned_elements: Role names that must appear as children.
        required_context_roles: Role names that must appear as ancestors.
        allowed_child_roles: Explicit set of allowed child roles (empty
            means no restriction on children).
        is_abstract: ``True`` for abstract roles that must not appear in
            authored content.
        name_from: How the accessible name is computed
            (``"author"``, ``"contents"``, ``"prohibited"``).
        is_presentational: ``True`` for ``presentation`` / ``none`` roles.
        description: Human-readable description from the specification.
    """

    name: str
    category: RoleCategory
    superclass_roles: FrozenSet[str] = field(default_factory=frozenset)
    subclass_roles: FrozenSet[str] = field(default_factory=frozenset)
    required_properties: FrozenSet[str] = field(default_factory=frozenset)
    supported_properties: FrozenSet[str] = field(default_factory=frozenset)
    required_states: FrozenSet[str] = field(default_factory=frozenset)
    supported_states: FrozenSet[str] = field(default_factory=frozenset)
    required_owned_elements: FrozenSet[str] = field(default_factory=frozenset)
    required_context_roles: FrozenSet[str] = field(default_factory=frozenset)
    allowed_child_roles: FrozenSet[str] = field(default_factory=frozenset)
    is_abstract: bool = False
    name_from: str = "author"
    is_presentational: bool = False
    description: str = ""

    @property
    def is_widget(self) -> bool:
        return self.category == RoleCategory.WIDGET

    @property
    def is_landmark(self) -> bool:
        return self.category == RoleCategory.LANDMARK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "superclass_roles": sorted(self.superclass_roles),
            "subclass_roles": sorted(self.subclass_roles),
            "required_properties": sorted(self.required_properties),
            "supported_properties": sorted(self.supported_properties),
            "required_states": sorted(self.required_states),
            "supported_states": sorted(self.supported_states),
            "required_owned_elements": sorted(self.required_owned_elements),
            "required_context_roles": sorted(self.required_context_roles),
            "allowed_child_roles": sorted(self.allowed_child_roles),
            "is_abstract": self.is_abstract,
            "name_from": self.name_from,
            "is_presentational": self.is_presentational,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AriaRole:
        return cls(
            name=str(d["name"]),
            category=RoleCategory(d["category"]),
            superclass_roles=frozenset(d.get("superclass_roles", [])),
            subclass_roles=frozenset(d.get("subclass_roles", [])),
            required_properties=frozenset(d.get("required_properties", [])),
            supported_properties=frozenset(d.get("supported_properties", [])),
            required_states=frozenset(d.get("required_states", [])),
            supported_states=frozenset(d.get("supported_states", [])),
            required_owned_elements=frozenset(d.get("required_owned_elements", [])),
            required_context_roles=frozenset(d.get("required_context_roles", [])),
            allowed_child_roles=frozenset(d.get("allowed_child_roles", [])),
            is_abstract=bool(d.get("is_abstract", False)),
            name_from=str(d.get("name_from", "author")),
            is_presentational=bool(d.get("is_presentational", False)),
            description=str(d.get("description", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# RoleRelationship
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RoleRelationship:
    """A directed relationship between two ARIA roles.

    Encodes parent/child, superclass/subclass, and context requirements
    from the taxonomy.

    Attributes:
        source_role: The role from which the relationship originates.
        target_role: The role to which the relationship points.
        relationship_type: One of ``"superclass"``, ``"subclass"``,
            ``"required_context"``, ``"required_owned"``, ``"allowed_child"``.
        description: Human-readable explanation.
    """

    source_role: str
    target_role: str
    relationship_type: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_role": self.source_role,
            "target_role": self.target_role,
            "relationship_type": self.relationship_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RoleRelationship:
        return cls(
            source_role=str(d["source_role"]),
            target_role=str(d["target_role"]),
            relationship_type=str(d["relationship_type"]),
            description=str(d.get("description", "")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# LandmarkRegion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class LandmarkRegion:
    """A detected ARIA landmark region in a page.

    Attributes:
        role: Landmark role name (``"banner"``, ``"main"``, etc.).
        label: Accessible name of the landmark (may be empty).
        node_id: Accessibility-tree node id of the landmark element.
        child_landmark_ids: Node ids of nested landmarks.
        contains_interactive: Whether the landmark contains interactive
            widgets.
    """

    role: str
    label: str
    node_id: str
    child_landmark_ids: Tuple[str, ...] = ()
    contains_interactive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "label": self.label,
            "node_id": self.node_id,
            "child_landmark_ids": list(self.child_landmark_ids),
            "contains_interactive": self.contains_interactive,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LandmarkRegion:
        return cls(
            role=str(d["role"]),
            label=str(d["label"]),
            node_id=str(d["node_id"]),
            child_landmark_ids=tuple(d.get("child_landmark_ids", [])),
            contains_interactive=bool(d.get("contains_interactive", False)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ConformanceResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ConformanceResult:
    """Result of ARIA conformance checking for a single element.

    Attributes:
        node_id: Accessibility-tree node identifier.
        role: ARIA role assigned to the element.
        level: Conformance level achieved.
        violations: Descriptions of conformance violations.
        warnings: Descriptions of non-critical conformance warnings.
        missing_properties: Required properties that are absent.
        missing_states: Required states that are absent.
        invalid_children: Child roles that violate ownership constraints.
        invalid_context: Whether the required context role is missing.
    """

    node_id: str
    role: str
    level: ConformanceLevel
    violations: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    missing_properties: FrozenSet[str] = field(default_factory=frozenset)
    missing_states: FrozenSet[str] = field(default_factory=frozenset)
    invalid_children: Tuple[str, ...] = ()
    invalid_context: bool = False

    @property
    def is_conforming(self) -> bool:
        """Whether the element fully conforms."""
        return self.level == ConformanceLevel.CONFORMING

    @property
    def num_issues(self) -> int:
        """Total number of violations and warnings."""
        return len(self.violations) + len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "level": self.level.value,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "missing_properties": sorted(self.missing_properties),
            "missing_states": sorted(self.missing_states),
            "invalid_children": list(self.invalid_children),
            "invalid_context": self.invalid_context,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConformanceResult:
        return cls(
            node_id=str(d["node_id"]),
            role=str(d["role"]),
            level=ConformanceLevel(d["level"]),
            violations=tuple(d.get("violations", [])),
            warnings=tuple(d.get("warnings", [])),
            missing_properties=frozenset(d.get("missing_properties", [])),
            missing_states=frozenset(d.get("missing_states", [])),
            invalid_children=tuple(d.get("invalid_children", [])),
            invalid_context=bool(d.get("invalid_context", False)),
        )
