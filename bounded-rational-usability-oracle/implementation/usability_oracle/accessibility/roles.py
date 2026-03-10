"""ARIA role taxonomy with hierarchy, classification, and semantic similarity."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional


class RoleTaxonomy:
    """Complete ARIA role taxonomy with hierarchy lookups and semantic distance."""

    # ── Role hierarchy: parent -> children ────────────────────────────────
    ROLE_HIERARCHY: dict[str, list[str]] = {
        "roletype": ["structure", "widget", "window"],
        # Structure roles
        "structure": [
            "application",
            "document",
            "presentation",
            "none",
            "section",
            "sectionhead",
            "separator",
            "range",
        ],
        "section": [
            "alert",
            "blockquote",
            "caption",
            "cell",
            "code",
            "definition",
            "deletion",
            "emphasis",
            "figure",
            "group",
            "img",
            "insertion",
            "landmark",
            "list",
            "listitem",
            "log",
            "marquee",
            "math",
            "note",
            "paragraph",
            "status",
            "strong",
            "subscript",
            "superscript",
            "table",
            "tabpanel",
            "term",
            "time",
            "tooltip",
        ],
        "sectionhead": [
            "columnheader",
            "heading",
            "rowheader",
            "tab",
        ],
        "landmark": [
            "banner",
            "complementary",
            "contentinfo",
            "form",
            "main",
            "navigation",
            "region",
            "search",
        ],
        "cell": ["columnheader", "gridcell", "rowheader"],
        "list": ["directory", "feed", "listbox", "menu", "tablist"],
        "group": ["row", "select", "toolbar"],
        "select": ["listbox", "menu", "radiogroup", "tree"],
        "range": ["meter", "progressbar", "scrollbar", "slider", "spinbutton"],
        # Widget roles
        "widget": ["command", "composite", "gridcell", "input", "separator"],
        "command": ["button", "link", "menuitem"],
        "composite": [
            "grid",
            "listbox",
            "menu",
            "menubar",
            "radiogroup",
            "select",
            "tablist",
            "tree",
            "treegrid",
        ],
        "input": ["checkbox", "combobox", "option", "radio", "slider", "spinbutton", "textbox"],
        "menuitem": ["menuitemcheckbox", "menuitemradio"],
        "checkbox": ["menuitemcheckbox", "switch"],
        "option": ["treeitem"],
        "grid": ["treegrid"],
        # Window roles
        "window": ["alertdialog", "dialog"],
    }

    # ── Classification sets ───────────────────────────────────────────────
    INTERACTIVE_ROLES: frozenset[str] = frozenset(
        {
            "button",
            "checkbox",
            "combobox",
            "grid",
            "gridcell",
            "link",
            "listbox",
            "menu",
            "menubar",
            "menuitem",
            "menuitemcheckbox",
            "menuitemradio",
            "option",
            "radio",
            "radiogroup",
            "scrollbar",
            "searchbox",
            "slider",
            "spinbutton",
            "switch",
            "tab",
            "tablist",
            "textbox",
            "tree",
            "treegrid",
            "treeitem",
        }
    )

    LANDMARK_ROLES: frozenset[str] = frozenset(
        {
            "banner",
            "complementary",
            "contentinfo",
            "form",
            "main",
            "navigation",
            "region",
            "search",
        }
    )

    WIDGET_ROLES: frozenset[str] = frozenset(
        {
            "button",
            "checkbox",
            "combobox",
            "grid",
            "gridcell",
            "link",
            "listbox",
            "menu",
            "menubar",
            "menuitem",
            "menuitemcheckbox",
            "menuitemradio",
            "option",
            "progressbar",
            "radio",
            "radiogroup",
            "scrollbar",
            "searchbox",
            "slider",
            "spinbutton",
            "switch",
            "tab",
            "tablist",
            "textbox",
            "tree",
            "treegrid",
            "treeitem",
            "alertdialog",
            "dialog",
            "tooltip",
        }
    )

    STRUCTURE_ROLES: frozenset[str] = frozenset(
        {
            "application",
            "article",
            "blockquote",
            "caption",
            "cell",
            "code",
            "columnheader",
            "definition",
            "deletion",
            "directory",
            "document",
            "emphasis",
            "feed",
            "figure",
            "group",
            "heading",
            "img",
            "insertion",
            "list",
            "listitem",
            "math",
            "none",
            "note",
            "paragraph",
            "presentation",
            "row",
            "rowgroup",
            "rowheader",
            "separator",
            "strong",
            "subscript",
            "superscript",
            "table",
            "term",
            "time",
            "toolbar",
        }
    )

    # Expected children mapping (role -> allowed child roles)
    EXPECTED_CHILDREN: dict[str, set[str]] = {
        "list": {"listitem"},
        "listbox": {"option"},
        "menu": {"menuitem", "menuitemcheckbox", "menuitemradio", "separator", "group"},
        "menubar": {"menuitem", "menuitemcheckbox", "menuitemradio"},
        "tablist": {"tab"},
        "tree": {"treeitem", "group"},
        "grid": {"row", "rowgroup"},
        "treegrid": {"row", "rowgroup"},
        "table": {"row", "rowgroup", "caption"},
        "row": {"cell", "columnheader", "gridcell", "rowheader"},
        "rowgroup": {"row"},
        "radiogroup": {"radio"},
        "select": {"option"},
        "toolbar": {"button", "checkbox", "link", "separator", "menuitem"},
        "feed": {"article"},
        "figure": {"img", "figcaption"},
    }

    def __init__(self) -> None:
        self._parent_map: dict[str, str] = {}
        self._depth_cache: dict[str, int] = {}
        self._all_roles: set[str] = set()
        self._build_parent_map()

    # ── Build helpers ─────────────────────────────────────────────────────

    def _build_parent_map(self) -> None:
        """Build reverse parent lookup from hierarchy."""
        for parent, children in self.ROLE_HIERARCHY.items():
            self._all_roles.add(parent)
            for child in children:
                self._all_roles.add(child)
                # Keep first-encountered parent (primary lineage)
                if child not in self._parent_map:
                    self._parent_map[child] = parent

    # ── Classification queries ────────────────────────────────────────────

    def is_interactive(self, role: str) -> bool:
        """Return True if the role is interactive (accepts user input)."""
        return role in self.INTERACTIVE_ROLES

    def is_landmark(self, role: str) -> bool:
        """Return True if the role is a landmark region."""
        return role in self.LANDMARK_ROLES

    def is_widget(self, role: str) -> bool:
        """Return True if the role is a widget."""
        return role in self.WIDGET_ROLES

    def is_structure(self, role: str) -> bool:
        """Return True if the role is a structural role."""
        return role in self.STRUCTURE_ROLES

    def is_valid_role(self, role: str) -> bool:
        """Return True if the role is known to the taxonomy."""
        return role in self._all_roles

    # ── Hierarchy queries ─────────────────────────────────────────────────

    def parent_role(self, role: str) -> Optional[str]:
        """Return the immediate parent role in the hierarchy, or None."""
        return self._parent_map.get(role)

    def ancestors(self, role: str) -> list[str]:
        """Return the chain of ancestor roles from immediate parent to root."""
        chain: list[str] = []
        current = self._parent_map.get(role)
        visited: set[str] = set()
        while current is not None and current not in visited:
            chain.append(current)
            visited.add(current)
            current = self._parent_map.get(current)
        return chain

    def depth(self, role: str) -> int:
        """Return depth of role in hierarchy (roletype=0)."""
        if role in self._depth_cache:
            return self._depth_cache[role]
        d = len(self.ancestors(role))
        self._depth_cache[role] = d
        return d

    def is_descendant_of(self, role: str, ancestor_role: str) -> bool:
        """Return True if *role* is a descendant of *ancestor_role*."""
        if role == ancestor_role:
            return True
        return ancestor_role in self.ancestors(role)

    def children(self, role: str) -> list[str]:
        """Return immediate child roles."""
        return list(self.ROLE_HIERARCHY.get(role, []))

    def descendants(self, role: str) -> set[str]:
        """Return all descendant roles (BFS)."""
        result: set[str] = set()
        queue = list(self.ROLE_HIERARCHY.get(role, []))
        while queue:
            current = queue.pop(0)
            if current not in result:
                result.add(current)
                queue.extend(self.ROLE_HIERARCHY.get(current, []))
        return result

    # ── Semantic similarity ───────────────────────────────────────────────

    @lru_cache(maxsize=1024)
    def _lca_role(self, role_a: str, role_b: str) -> str:
        """Find lowest common ancestor of two roles in the hierarchy."""
        ancestors_a = [role_a] + self.ancestors(role_a)
        ancestors_b_set = set([role_b] + self.ancestors(role_b))
        for anc in ancestors_a:
            if anc in ancestors_b_set:
                return anc
        return "roletype"

    def semantic_similarity(self, role_a: str, role_b: str) -> float:
        """Compute semantic similarity in [0, 1] based on taxonomy distance.

        Uses the Wu-Palmer similarity measure:
            sim = 2 * depth(LCA) / (depth(a) + depth(b))
        Returns 1.0 for identical roles, lower for distant roles.
        """
        if role_a == role_b:
            return 1.0

        # Unknown roles get 0 similarity
        if role_a not in self._all_roles or role_b not in self._all_roles:
            return 0.0

        lca = self._lca_role(role_a, role_b)
        depth_lca = self.depth(lca) + 1  # +1 to avoid zero
        depth_a = self.depth(role_a) + 1
        depth_b = self.depth(role_b) + 1

        return (2.0 * depth_lca) / (depth_a + depth_b)

    def taxonomy_distance(self, role_a: str, role_b: str) -> int:
        """Compute shortest path distance through the hierarchy tree."""
        if role_a == role_b:
            return 0
        lca = self._lca_role(role_a, role_b)
        return (self.depth(role_a) - self.depth(lca)) + (self.depth(role_b) - self.depth(lca))

    # ── Containment queries ───────────────────────────────────────────────

    def expected_children(self, role: str) -> set[str]:
        """Return the set of expected child roles, or empty set."""
        return set(self.EXPECTED_CHILDREN.get(role, set()))

    def can_contain(self, parent_role: str, child_role: str) -> bool:
        """Return True if *parent_role* may contain *child_role*.

        Returns True when there is no restriction (no expected_children entry)
        or when child_role is in the expected set.
        """
        expected = self.EXPECTED_CHILDREN.get(parent_role)
        if expected is None:
            return True  # no restriction
        return child_role in expected

    # ── Utilities ─────────────────────────────────────────────────────────

    def all_roles(self) -> frozenset[str]:
        """Return the set of all known roles."""
        return frozenset(self._all_roles)

    def role_category(self, role: str) -> str:
        """Classify a role into 'widget', 'landmark', 'structure', or 'other'."""
        if role in self.WIDGET_ROLES:
            return "widget"
        if role in self.LANDMARK_ROLES:
            return "landmark"
        if role in self.STRUCTURE_ROLES:
            return "structure"
        return "other"

    def __repr__(self) -> str:
        return f"RoleTaxonomy(roles={len(self._all_roles)})"


# Module-level singleton for convenience
_taxonomy = RoleTaxonomy()

is_interactive = _taxonomy.is_interactive
is_landmark = _taxonomy.is_landmark
parent_role = _taxonomy.parent_role
is_descendant_of = _taxonomy.is_descendant_of
semantic_similarity = _taxonomy.semantic_similarity
expected_children = _taxonomy.expected_children
can_contain = _taxonomy.can_contain
