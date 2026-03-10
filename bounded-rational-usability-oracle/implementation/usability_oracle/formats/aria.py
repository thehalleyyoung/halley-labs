"""
usability_oracle.formats.aria — Full ARIA specification parsing.

Parses WAI-ARIA 1.2 role, state, and property definitions from
structured data and validates accessibility trees against the
ARIA specification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# ARIA role taxonomy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ARIARole:
    """Specification of a single ARIA role."""
    name: str
    superclass: list[str] = field(default_factory=list)
    required_states: list[str] = field(default_factory=list)
    supported_states: list[str] = field(default_factory=list)
    required_owned: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    name_from: list[str] = field(default_factory=list)
    children_presentational: bool = False
    is_abstract: bool = False
    is_widget: bool = False
    is_landmark: bool = False


# Complete ARIA 1.2 role definitions
_ARIA_ROLES: dict[str, ARIARole] = {
    "alert": ARIARole(
        name="alert", superclass=["section"], is_widget=False,
        supported_states=["aria-atomic", "aria-busy", "aria-live"],
        name_from=["author"],
    ),
    "alertdialog": ARIARole(
        name="alertdialog", superclass=["alert", "dialog"],
        required_states=[], supported_states=["aria-modal"],
        name_from=["author"],
    ),
    "application": ARIARole(
        name="application", superclass=["structure"],
        name_from=["author"],
    ),
    "article": ARIARole(
        name="article", superclass=["document"],
        name_from=["author"],
    ),
    "banner": ARIARole(
        name="banner", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "button": ARIARole(
        name="button", superclass=["command"], is_widget=True,
        supported_states=["aria-disabled", "aria-expanded", "aria-haspopup", "aria-pressed"],
        name_from=["contents", "author"], children_presentational=True,
    ),
    "cell": ARIARole(
        name="cell", superclass=["section"],
        required_context=["row"],
        supported_states=["aria-colindex", "aria-colspan", "aria-rowindex", "aria-rowspan"],
        name_from=["contents", "author"],
    ),
    "checkbox": ARIARole(
        name="checkbox", superclass=["input"], is_widget=True,
        required_states=["aria-checked"],
        supported_states=["aria-readonly", "aria-required"],
        name_from=["contents", "author"], children_presentational=True,
    ),
    "columnheader": ARIARole(
        name="columnheader", superclass=["cell", "gridcell", "sectionhead"],
        required_context=["row"],
        supported_states=["aria-sort"],
        name_from=["contents", "author"],
    ),
    "combobox": ARIARole(
        name="combobox", superclass=["input"], is_widget=True,
        required_states=["aria-expanded"],
        required_owned=["listbox", "tree", "grid", "dialog"],
        supported_states=["aria-autocomplete", "aria-readonly", "aria-required"],
        name_from=["author"],
    ),
    "complementary": ARIARole(
        name="complementary", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "contentinfo": ARIARole(
        name="contentinfo", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "definition": ARIARole(
        name="definition", superclass=["section"],
        name_from=["author"],
    ),
    "dialog": ARIARole(
        name="dialog", superclass=["window"],
        supported_states=["aria-modal"],
        name_from=["author"],
    ),
    "directory": ARIARole(
        name="directory", superclass=["list"],
        name_from=["author"],
    ),
    "document": ARIARole(
        name="document", superclass=["structure"],
        name_from=["author"],
    ),
    "feed": ARIARole(
        name="feed", superclass=["list"],
        required_owned=["article"],
        name_from=["author"],
    ),
    "figure": ARIARole(
        name="figure", superclass=["section"],
        name_from=["author"],
    ),
    "form": ARIARole(
        name="form", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "grid": ARIARole(
        name="grid", superclass=["composite", "table"], is_widget=True,
        required_owned=["row", "rowgroup"],
        supported_states=["aria-multiselectable", "aria-readonly"],
        name_from=["author"],
    ),
    "gridcell": ARIARole(
        name="gridcell", superclass=["cell", "widget"], is_widget=True,
        required_context=["row"],
        supported_states=["aria-disabled", "aria-expanded", "aria-haspopup",
                          "aria-readonly", "aria-required", "aria-selected"],
        name_from=["contents", "author"],
    ),
    "group": ARIARole(
        name="group", superclass=["section"],
        name_from=["author"],
    ),
    "heading": ARIARole(
        name="heading", superclass=["sectionhead"],
        required_states=["aria-level"],
        name_from=["contents", "author"],
    ),
    "img": ARIARole(
        name="img", superclass=["section"],
        name_from=["author"], children_presentational=True,
    ),
    "link": ARIARole(
        name="link", superclass=["command"], is_widget=True,
        supported_states=["aria-disabled", "aria-expanded"],
        name_from=["contents", "author"],
    ),
    "list": ARIARole(
        name="list", superclass=["section"],
        required_owned=["listitem"],
        name_from=["author"],
    ),
    "listbox": ARIARole(
        name="listbox", superclass=["select"], is_widget=True,
        required_owned=["option"],
        supported_states=["aria-multiselectable", "aria-readonly", "aria-required"],
        name_from=["author"],
    ),
    "listitem": ARIARole(
        name="listitem", superclass=["section"],
        required_context=["list"],
        supported_states=["aria-level", "aria-posinset", "aria-setsize"],
        name_from=["author"],
    ),
    "log": ARIARole(
        name="log", superclass=["section"],
        name_from=["author"],
    ),
    "main": ARIARole(
        name="main", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "marquee": ARIARole(
        name="marquee", superclass=["section"],
        name_from=["author"],
    ),
    "math": ARIARole(
        name="math", superclass=["section"],
        name_from=["author"], children_presentational=True,
    ),
    "menu": ARIARole(
        name="menu", superclass=["select"], is_widget=True,
        required_owned=["menuitem", "menuitemcheckbox", "menuitemradio"],
        name_from=["author"],
    ),
    "menubar": ARIARole(
        name="menubar", superclass=["menu"], is_widget=True,
        required_owned=["menuitem", "menuitemcheckbox", "menuitemradio"],
        name_from=["author"],
    ),
    "menuitem": ARIARole(
        name="menuitem", superclass=["command"], is_widget=True,
        required_context=["menu", "menubar"],
        supported_states=["aria-disabled", "aria-expanded", "aria-haspopup",
                          "aria-posinset", "aria-setsize"],
        name_from=["contents", "author"],
    ),
    "navigation": ARIARole(
        name="navigation", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "none": ARIARole(name="none", superclass=["structure"], is_abstract=False),
    "note": ARIARole(
        name="note", superclass=["section"],
        name_from=["author"],
    ),
    "option": ARIARole(
        name="option", superclass=["input"], is_widget=True,
        required_states=["aria-selected"],
        required_context=["listbox"],
        name_from=["contents", "author"], children_presentational=True,
    ),
    "presentation": ARIARole(name="presentation", superclass=["structure"]),
    "progressbar": ARIARole(
        name="progressbar", superclass=["range"], is_widget=True,
        name_from=["author"], children_presentational=True,
    ),
    "radio": ARIARole(
        name="radio", superclass=["input"], is_widget=True,
        required_states=["aria-checked"],
        name_from=["contents", "author"], children_presentational=True,
    ),
    "radiogroup": ARIARole(
        name="radiogroup", superclass=["select"], is_widget=True,
        required_owned=["radio"],
        name_from=["author"],
    ),
    "region": ARIARole(
        name="region", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "row": ARIARole(
        name="row", superclass=["group", "widget"],
        required_context=["grid", "rowgroup", "table", "treegrid"],
        required_owned=["cell", "columnheader", "gridcell", "rowheader"],
        name_from=["contents", "author"],
    ),
    "rowgroup": ARIARole(
        name="rowgroup", superclass=["structure"],
        required_context=["grid", "table", "treegrid"],
        required_owned=["row"],
        name_from=["author"],
    ),
    "rowheader": ARIARole(
        name="rowheader", superclass=["cell", "gridcell", "sectionhead"],
        required_context=["row"],
        name_from=["contents", "author"],
    ),
    "scrollbar": ARIARole(
        name="scrollbar", superclass=["range"], is_widget=True,
        required_states=["aria-controls", "aria-valuenow"],
        name_from=["author"], children_presentational=True,
    ),
    "search": ARIARole(
        name="search", superclass=["landmark"], is_landmark=True,
        name_from=["author"],
    ),
    "searchbox": ARIARole(
        name="searchbox", superclass=["textbox"], is_widget=True,
        name_from=["author"],
    ),
    "separator": ARIARole(
        name="separator", superclass=["structure"],
        name_from=["author"], children_presentational=True,
    ),
    "slider": ARIARole(
        name="slider", superclass=["input", "range"], is_widget=True,
        required_states=["aria-valuenow"],
        supported_states=["aria-orientation", "aria-readonly", "aria-valuemax",
                          "aria-valuemin", "aria-valuetext"],
        name_from=["author"], children_presentational=True,
    ),
    "spinbutton": ARIARole(
        name="spinbutton", superclass=["composite", "input", "range"], is_widget=True,
        required_states=["aria-valuenow"],
        name_from=["author"],
    ),
    "status": ARIARole(
        name="status", superclass=["section"],
        name_from=["author"],
    ),
    "switch": ARIARole(
        name="switch", superclass=["checkbox"], is_widget=True,
        required_states=["aria-checked"],
        name_from=["contents", "author"], children_presentational=True,
    ),
    "tab": ARIARole(
        name="tab", superclass=["sectionhead", "widget"], is_widget=True,
        required_context=["tablist"],
        supported_states=["aria-disabled", "aria-expanded", "aria-haspopup", "aria-selected"],
        name_from=["contents", "author"],
    ),
    "table": ARIARole(
        name="table", superclass=["section"],
        required_owned=["row", "rowgroup"],
        name_from=["author"],
    ),
    "tablist": ARIARole(
        name="tablist", superclass=["composite"], is_widget=True,
        required_owned=["tab"],
        supported_states=["aria-multiselectable", "aria-orientation"],
        name_from=["author"],
    ),
    "tabpanel": ARIARole(
        name="tabpanel", superclass=["section"],
        name_from=["author"],
    ),
    "term": ARIARole(name="term", superclass=["section"], name_from=["author"]),
    "textbox": ARIARole(
        name="textbox", superclass=["input"], is_widget=True,
        supported_states=["aria-activedescendant", "aria-autocomplete", "aria-multiline",
                          "aria-placeholder", "aria-readonly", "aria-required"],
        name_from=["author"],
    ),
    "timer": ARIARole(
        name="timer", superclass=["status"],
        name_from=["author"],
    ),
    "toolbar": ARIARole(
        name="toolbar", superclass=["group"],
        supported_states=["aria-orientation"],
        name_from=["author"],
    ),
    "tooltip": ARIARole(
        name="tooltip", superclass=["section"],
        name_from=["contents", "author"],
    ),
    "tree": ARIARole(
        name="tree", superclass=["select"], is_widget=True,
        required_owned=["treeitem"],
        supported_states=["aria-multiselectable", "aria-required"],
        name_from=["author"],
    ),
    "treegrid": ARIARole(
        name="treegrid", superclass=["grid", "tree"], is_widget=True,
        required_owned=["row", "rowgroup"],
        name_from=["author"],
    ),
    "treeitem": ARIARole(
        name="treeitem", superclass=["listitem", "option"], is_widget=True,
        required_context=["group", "tree"],
        name_from=["contents", "author"],
    ),
}


@dataclass
class ARIAValidationIssue:
    """A single ARIA validation issue."""
    node_id: str
    role: str
    issue_type: str
    message: str
    severity: str = "warning"


# ---------------------------------------------------------------------------
# ARIAParser
# ---------------------------------------------------------------------------

class ARIAParser:
    """Parse and validate ARIA specifications."""

    def __init__(self) -> None:
        self._roles = _ARIA_ROLES

    def get_role(self, name: str) -> Optional[ARIARole]:
        """Look up an ARIA role by name."""
        return self._roles.get(name.lower())

    def is_valid_role(self, name: str) -> bool:
        """Check if a role name is a valid ARIA role."""
        return name.lower() in self._roles

    def is_widget_role(self, name: str) -> bool:
        """Check if a role is a widget (interactive)."""
        role = self._roles.get(name.lower())
        return role.is_widget if role else False

    def is_landmark_role(self, name: str) -> bool:
        """Check if a role is a landmark."""
        role = self._roles.get(name.lower())
        return role.is_landmark if role else False

    def required_states(self, role_name: str) -> list[str]:
        """Get required ARIA states for a role."""
        role = self._roles.get(role_name.lower())
        return list(role.required_states) if role else []

    def supported_states(self, role_name: str) -> list[str]:
        """Get all supported ARIA states for a role."""
        role = self._roles.get(role_name.lower())
        return list(role.supported_states) if role else []

    def required_context(self, role_name: str) -> list[str]:
        """Get required parent context roles."""
        role = self._roles.get(role_name.lower())
        return list(role.required_context) if role else []

    def required_owned(self, role_name: str) -> list[str]:
        """Get required child element roles."""
        role = self._roles.get(role_name.lower())
        return list(role.required_owned) if role else []

    def validate_tree(self, tree: Any) -> list[ARIAValidationIssue]:
        """Validate an accessibility tree against ARIA spec.

        Checks:
        - Valid role names
        - Required states present
        - Required context (parent roles)
        - Required owned elements (child roles)
        - Name computation requirements
        """
        issues: list[ARIAValidationIssue] = []
        self._validate_node(tree.root, None, issues)
        return issues

    def _validate_node(
        self,
        node: Any,
        parent_role: Optional[str],
        issues: list[ARIAValidationIssue],
    ) -> None:
        role_name = node.role.lower() if isinstance(node.role, str) else str(node.role).lower()
        role_spec = self._roles.get(role_name)

        if not role_spec and role_name not in ("generic", "none", "presentation"):
            issues.append(ARIAValidationIssue(
                node_id=node.id, role=role_name,
                issue_type="invalid_role",
                message=f"Unknown ARIA role: {role_name}",
            ))
        elif role_spec:
            # Check required states
            props = getattr(node, "properties", {}) or {}
            for required in role_spec.required_states:
                if required not in props:
                    issues.append(ARIAValidationIssue(
                        node_id=node.id, role=role_name,
                        issue_type="missing_required_state",
                        message=f"Missing required state '{required}' for role '{role_name}'",
                    ))

            # Check required context
            if role_spec.required_context and parent_role:
                if parent_role.lower() not in [r.lower() for r in role_spec.required_context]:
                    issues.append(ARIAValidationIssue(
                        node_id=node.id, role=role_name,
                        issue_type="invalid_context",
                        message=f"Role '{role_name}' requires parent context {role_spec.required_context}, found '{parent_role}'",
                        severity="error",
                    ))

            # Check required owned elements
            if role_spec.required_owned and node.children:
                child_roles = {
                    (c.role.lower() if isinstance(c.role, str) else str(c.role).lower())
                    for c in node.children
                }
                required_set = {r.lower() for r in role_spec.required_owned}
                if not child_roles & required_set:
                    issues.append(ARIAValidationIssue(
                        node_id=node.id, role=role_name,
                        issue_type="missing_required_owned",
                        message=f"Role '{role_name}' requires owned elements: {role_spec.required_owned}",
                    ))

            # Check accessible name
            name = getattr(node, "name", "")
            if not name and "author" in role_spec.name_from and role_spec.is_widget:
                issues.append(ARIAValidationIssue(
                    node_id=node.id, role=role_name,
                    issue_type="missing_accessible_name",
                    message=f"Widget role '{role_name}' has no accessible name",
                    severity="error",
                ))

        for child in node.children:
            self._validate_node(child, role_name, issues)

    def list_all_roles(self) -> list[str]:
        """Return all known ARIA role names."""
        return sorted(self._roles.keys())

    def list_widget_roles(self) -> list[str]:
        """Return all widget role names."""
        return sorted(name for name, role in self._roles.items() if role.is_widget)

    def list_landmark_roles(self) -> list[str]:
        """Return all landmark role names."""
        return sorted(name for name, role in self._roles.items() if role.is_landmark)
