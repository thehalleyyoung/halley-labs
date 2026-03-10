"""
usability_oracle.android_a11y — Android accessibility format support.

Provides types and protocols for parsing Android view hierarchy dumps
(``uiautomator dump`` XML and JSON formats) and normalising them into
the oracle's common accessibility-tree representation.

::

    from usability_oracle.android_a11y import AndroidNode, ViewHierarchy
    from usability_oracle.android_a11y import AndroidAccessibilityParser
    from usability_oracle.android_a11y import AndroidHierarchyNormalizer
"""

from __future__ import annotations

from usability_oracle.android_a11y.types import (
    AccessibilityAction,
    AccessibilityActionId,
    AndroidClassName,
    AndroidNode,
    BoundsInfo,
    ContentDescription,
    ViewHierarchy,
)

from usability_oracle.android_a11y.protocols import (
    AndroidParser,
    HierarchyNormalizer,
)

from usability_oracle.android_a11y.parser import (
    AndroidAccessibilityParser,
)

from usability_oracle.android_a11y.normalizer import (
    AndroidHierarchyNormalizer,
)

from usability_oracle.android_a11y.converter import (
    AndroidToAccessibilityConverter,
)

from usability_oracle.android_a11y import conformance as android_conformance

__all__ = [
    # types
    "AccessibilityAction",
    "AccessibilityActionId",
    "AndroidClassName",
    "AndroidNode",
    "BoundsInfo",
    "ContentDescription",
    "ViewHierarchy",
    # protocols
    "AndroidParser",
    "HierarchyNormalizer",
    # parser
    "AndroidAccessibilityParser",
    # normalizer
    "AndroidHierarchyNormalizer",
    # converter
    "AndroidToAccessibilityConverter",
    # conformance
    "android_conformance",
]
