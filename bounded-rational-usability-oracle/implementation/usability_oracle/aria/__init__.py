"""
usability_oracle.aria — WAI-ARIA 1.2 taxonomy and conformance checking.

Provides the complete role taxonomy (82 roles), property/state definitions,
and conformance checking utilities per the WAI-ARIA 1.2 specification.

::

    from usability_oracle.aria import AriaRole, ROLE_TAXONOMY, get_role
    from usability_oracle.aria import AriaHTMLParser, AriaConformanceChecker
"""

from __future__ import annotations

from usability_oracle.aria.types import (
    AriaProperty,
    AriaRole,
    AriaState,
    ConformanceLevel,
    ConformanceResult,
    LandmarkRegion,
    PropertyType,
    RoleCategory,
    RoleRelationship,
)

from usability_oracle.aria.taxonomy import (
    ROLE_TAXONOMY,
    get_role,
    is_superclass_of,
    role_names,
    roles_by_category,
)

from usability_oracle.aria.parser import (
    AriaHTMLParser,
    AriaNodeInfo,
    AriaTree,
)

from usability_oracle.aria.conformance import (
    AriaConformanceChecker,
)

from usability_oracle.aria.converter import (
    AriaToAccessibilityConverter,
)

from usability_oracle.aria.validator import (
    validate_aria_document,
    ValidationReport,
    ValidationIssue,
)

__all__ = [
    # types
    "AriaProperty",
    "AriaRole",
    "AriaState",
    "ConformanceLevel",
    "ConformanceResult",
    "LandmarkRegion",
    "PropertyType",
    "RoleCategory",
    "RoleRelationship",
    # taxonomy
    "ROLE_TAXONOMY",
    "get_role",
    "is_superclass_of",
    "role_names",
    "roles_by_category",
    # parser
    "AriaHTMLParser",
    "AriaNodeInfo",
    "AriaTree",
    # conformance
    "AriaConformanceChecker",
    # converter
    "AriaToAccessibilityConverter",
    # validator
    "validate_aria_document",
    "ValidationReport",
    "ValidationIssue",
]
