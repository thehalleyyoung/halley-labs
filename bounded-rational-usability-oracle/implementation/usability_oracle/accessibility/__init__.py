"""Accessibility module — parse, normalise, and analyse UI accessibility trees."""

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityProperty,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.json_parser import JSONAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.accessibility.normalizer import (
    AccessibilityNormalizer as Normalizer,
)
from usability_oracle.accessibility.roles import RoleTaxonomy
from usability_oracle.accessibility.spatial import (
    LayoutInfo,
    NodeGroup,
    SpatialAnalyzer,
)
from usability_oracle.accessibility.validators import (
    TreeValidator,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    # Models
    "AccessibilityNode",
    "AccessibilityProperty",
    "AccessibilityState",
    "AccessibilityTree",
    "BoundingBox",
    # Parsers
    "HTMLAccessibilityParser",
    "JSONAccessibilityParser",
    # Normaliser
    "AccessibilityNormalizer",
    "Normalizer",
    # Roles
    "RoleTaxonomy",
    # Spatial
    "LayoutInfo",
    "NodeGroup",
    "SpatialAnalyzer",
    # Validation
    "TreeValidator",
    "ValidationIssue",
    "ValidationResult",
]
