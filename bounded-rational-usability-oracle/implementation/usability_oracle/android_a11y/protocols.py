"""
usability_oracle.android_a11y.protocols — Structural interfaces for
Android accessibility format parsing and normalisation.

Defines protocols for parsing Android view hierarchy dumps and
normalising them into the oracle's common accessibility-tree
representation.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.android_a11y.types import (
        AndroidNode,
        ViewHierarchy,
    )


# ═══════════════════════════════════════════════════════════════════════════
# AndroidParser
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class AndroidParser(Protocol):
    """Parse raw Android accessibility dumps into structured hierarchies.

    Supports both XML (``uiautomator dump``) and JSON formats
    (Accessibility Service output, Rico dataset).
    """

    def parse_xml(self, xml_content: str) -> ViewHierarchy:
        """Parse a ``uiautomator dump`` XML string.

        Parameters:
            xml_content: Raw XML output from ``uiautomator dump``.

        Returns:
            Parsed :class:`ViewHierarchy`.

        Raises:
            ParseError: On malformed or empty XML.
        """
        ...

    def parse_json(self, json_content: str) -> ViewHierarchy:
        """Parse a JSON-format view hierarchy (e.g. Rico dataset).

        Parameters:
            json_content: Raw JSON string representing the hierarchy.

        Returns:
            Parsed :class:`ViewHierarchy`.

        Raises:
            ParseError: On malformed or empty JSON.
        """
        ...

    def parse_dict(self, data: Dict[str, Any]) -> ViewHierarchy:
        """Parse an already-deserialised dictionary.

        Parameters:
            data: Dictionary representation of the hierarchy.

        Returns:
            Parsed :class:`ViewHierarchy`.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# HierarchyNormalizer
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class HierarchyNormalizer(Protocol):
    """Normalise an Android view hierarchy into the oracle's common format.

    The normaliser maps Android-specific class names, states, and
    semantics to the platform-agnostic accessibility tree used by the
    rest of the pipeline.
    """

    def normalize(
        self,
        hierarchy: ViewHierarchy,
    ) -> Dict[str, Any]:
        """Convert an Android hierarchy to a normalised accessibility tree.

        The output is a serialised accessibility tree compatible with
        the oracle's :class:`Parser` protocol output.

        Parameters:
            hierarchy: Parsed Android :class:`ViewHierarchy`.

        Returns:
            Serialised accessibility tree (dict) in the oracle's
            common format.
        """
        ...

    def map_class_to_role(self, class_name: str) -> str:
        """Map an Android widget class name to a WAI-ARIA role.

        Parameters:
            class_name: Fully-qualified Android class name
                (e.g. ``"android.widget.Button"``).

        Returns:
            ARIA role name (e.g. ``"button"``).
        """
        ...

    def filter_important_nodes(
        self,
        hierarchy: ViewHierarchy,
    ) -> Sequence[AndroidNode]:
        """Return only nodes that are important for accessibility.

        Removes decorative containers, invisible nodes, and layout-only
        groups that do not contribute to the accessible interface.

        Parameters:
            hierarchy: Full :class:`ViewHierarchy`.

        Returns:
            Filtered sequence of :class:`AndroidNode` instances.
        """
        ...
