"""
usability_oracle.aria.conformance — ARIA conformance checking.

Validates ARIA role usage against the WAI-ARIA 1.2 specification,
checking required properties, allowed properties, parent/child
constraints, accessible name requirements, focusability, and landmark
structure.

Reference: https://www.w3.org/TR/wai-aria-1.2/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

from usability_oracle.aria.types import (
    AriaRole,
    ConformanceLevel,
    ConformanceResult,
    LandmarkRegion,
    RoleCategory,
)
from usability_oracle.aria.taxonomy import ROLE_TAXONOMY, get_role, is_superclass_of
from usability_oracle.aria.parser import AriaNodeInfo, AriaTree


# ═══════════════════════════════════════════════════════════════════════════
# Global properties — applicable to every role (WAI-ARIA 1.2 §6.6)
# ═══════════════════════════════════════════════════════════════════════════

_GLOBAL_PROPERTIES: FrozenSet[str] = frozenset({
    "atomic", "busy", "controls", "current", "describedby",
    "details", "disabled", "dropeffect", "errormessage",
    "flowto", "grabbed", "haspopup", "hidden", "invalid",
    "keyshortcuts", "label", "labelledby", "live", "owns",
    "relevant", "roledescription",
})

_GLOBAL_STATES: FrozenSet[str] = frozenset({
    "disabled", "hidden", "invalid",
})

_INTERACTIVE_ROLES: FrozenSet[str] = frozenset({
    "button", "checkbox", "combobox", "gridcell", "link", "listbox",
    "menu", "menubar", "menuitem", "menuitemcheckbox", "menuitemradio",
    "option", "radio", "scrollbar", "searchbox", "slider", "spinbutton",
    "switch", "tab", "textbox", "treeitem",
})

_LANDMARK_ROLES: FrozenSet[str] = frozenset({
    "banner", "complementary", "contentinfo", "form",
    "main", "navigation", "region", "search",
})

# Roles that require an accessible name (WAI-ARIA 1.2 §5.2.8)
_NAME_REQUIRED_ROLES: FrozenSet[str] = frozenset({
    "checkbox", "combobox", "heading", "link", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "radio",
    "slider", "spinbutton", "switch", "tab", "textbox",
    "treeitem", "button", "meter", "progressbar",
    "form", "region",
})


# ═══════════════════════════════════════════════════════════════════════════
# AriaConformanceChecker
# ═══════════════════════════════════════════════════════════════════════════

class AriaConformanceChecker:
    """Check ARIA conformance of an :class:`AriaTree`.

    Performs per-node and whole-tree conformance checks against the
    WAI-ARIA 1.2 specification.

    Usage::

        checker = AriaConformanceChecker()
        results = checker.run_all_checks(aria_tree)
        for r in results:
            if not r.is_conforming:
                print(r.node_id, r.violations)
    """

    # ── Per-node checks ───────────────────────────────────────────────────

    def check_required_properties(self, node: AriaNodeInfo) -> ConformanceResult:
        """Check that the node's role has all required aria-* properties.

        WAI-ARIA 1.2 §5.2.5: *"Required States and Properties:
        attributes that are essential … The user agent SHOULD inform
        authors of missing required states and properties."*

        Parameters:
            node: An :class:`AriaNodeInfo` to validate.

        Returns:
            :class:`ConformanceResult` recording any missing properties.
        """
        role_def = get_role(node.role)
        if role_def is None:
            return ConformanceResult(
                node_id=node.node_id,
                role=node.role,
                level=ConformanceLevel.WARNING,
                warnings=(f"Unknown role '{node.role}'",),
            )

        present = set(node.properties.keys()) | set(node.states.keys())
        missing_props = role_def.required_properties - present
        missing_states = role_def.required_states - present

        violations: list[str] = []
        if missing_props:
            violations.append(
                f"Missing required properties for role '{node.role}': "
                f"{', '.join(sorted(missing_props))} (§5.2.5)"
            )
        if missing_states:
            violations.append(
                f"Missing required states for role '{node.role}': "
                f"{', '.join(sorted(missing_states))} (§5.2.5)"
            )

        level = ConformanceLevel.VIOLATION if violations else ConformanceLevel.CONFORMING
        return ConformanceResult(
            node_id=node.node_id,
            role=node.role,
            level=level,
            violations=tuple(violations),
            missing_properties=frozenset(missing_props),
            missing_states=frozenset(missing_states),
        )

    def check_allowed_properties(self, node: AriaNodeInfo) -> ConformanceResult:
        """Check that no disallowed properties are present for the role.

        WAI-ARIA 1.2 §6.6: Properties not listed as supported or
        required for a role, and not global, should not be used.

        Parameters:
            node: An :class:`AriaNodeInfo` to validate.

        Returns:
            :class:`ConformanceResult` with any disallowed property warnings.
        """
        role_def = get_role(node.role)
        if role_def is None:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        allowed = (
            role_def.required_properties
            | role_def.supported_properties
            | role_def.required_states
            | role_def.supported_states
            | _GLOBAL_PROPERTIES
            | _GLOBAL_STATES
        )

        present = set(node.properties.keys()) | set(node.states.keys())
        disallowed = present - allowed
        # Remove common non-property attrs that might be in properties dict
        disallowed -= {"describedby_text"}

        warnings: list[str] = []
        if disallowed:
            warnings.append(
                f"Properties not allowed on role '{node.role}': "
                f"{', '.join(sorted(disallowed))} (§6.6)"
            )

        level = ConformanceLevel.WARNING if warnings else ConformanceLevel.CONFORMING
        return ConformanceResult(
            node_id=node.node_id, role=node.role,
            level=level, warnings=tuple(warnings),
        )

    def check_required_children(self, node: AriaNodeInfo) -> ConformanceResult:
        """Check that required owned elements are present as children.

        WAI-ARIA 1.2 §5.2.6: *"Required Owned Elements: any element
        with this role must contain or own at least one of the listed
        child roles."*

        Parameters:
            node: An :class:`AriaNodeInfo` to validate.

        Returns:
            :class:`ConformanceResult` with violations for missing children.
        """
        role_def = get_role(node.role)
        if role_def is None or not role_def.required_owned_elements:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        child_roles = self._collect_child_roles(node)

        # Check if at least one required owned role is present
        has_required = any(
            any(
                cr == req or is_superclass_of(req, cr)
                for cr in child_roles
            )
            for req in role_def.required_owned_elements
        )

        if has_required:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        return ConformanceResult(
            node_id=node.node_id, role=node.role,
            level=ConformanceLevel.VIOLATION,
            violations=(
                f"Role '{node.role}' requires owned elements with role(s) "
                f"{', '.join(sorted(role_def.required_owned_elements))}, "
                f"but found: {', '.join(sorted(child_roles)) or 'none'} (§5.2.6)",
            ),
            invalid_children=tuple(sorted(child_roles)),
        )

    def check_required_parent(self, node: AriaNodeInfo, tree: AriaTree) -> ConformanceResult:
        """Check that the node appears within a required context role.

        WAI-ARIA 1.2 §5.2.7: *"Required Context Role: the element with
        this role must be contained within … an element with the listed
        role."*

        Parameters:
            node: An :class:`AriaNodeInfo` to validate.
            tree: The full :class:`AriaTree` for ancestor lookup.

        Returns:
            :class:`ConformanceResult` with violations if context is missing.
        """
        role_def = get_role(node.role)
        if role_def is None or not role_def.required_context_roles:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        # Walk ancestors
        ancestor_roles = self._collect_ancestor_roles(node, tree)

        has_context = any(
            any(
                ar == ctx or is_superclass_of(ctx, ar)
                for ar in ancestor_roles
            )
            for ctx in role_def.required_context_roles
        )

        if has_context:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        return ConformanceResult(
            node_id=node.node_id, role=node.role,
            level=ConformanceLevel.VIOLATION,
            violations=(
                f"Role '{node.role}' requires context role(s) "
                f"{', '.join(sorted(role_def.required_context_roles))}, "
                f"but found ancestors: "
                f"{', '.join(sorted(ancestor_roles)) or 'none'} (§5.2.7)",
            ),
            invalid_context=True,
        )

    def check_name_required(self, node: AriaNodeInfo) -> ConformanceResult:
        """Check that roles requiring an accessible name have one.

        WAI-ARIA 1.2 §5.2.8: Certain roles *require* an accessible name.

        Parameters:
            node: An :class:`AriaNodeInfo` to validate.

        Returns:
            :class:`ConformanceResult` with a violation if name is missing.
        """
        if node.role not in _NAME_REQUIRED_ROLES:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        if node.accessible_name and node.accessible_name.strip():
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        return ConformanceResult(
            node_id=node.node_id, role=node.role,
            level=ConformanceLevel.VIOLATION,
            violations=(
                f"Role '{node.role}' requires an accessible name but "
                f"none was found (§5.2.8)",
            ),
        )

    def check_focusable(self, node: AriaNodeInfo) -> ConformanceResult:
        """Check that interactive roles are keyboard-focusable.

        WAI-ARIA 1.2 §6.5: Interactive widgets must be reachable via
        the keyboard (either natively focusable or via tabindex).

        Parameters:
            node: An :class:`AriaNodeInfo` to validate.

        Returns:
            :class:`ConformanceResult` with warning if not focusable.
        """
        if node.role not in _INTERACTIVE_ROLES:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        if node.is_focusable:
            return ConformanceResult(
                node_id=node.node_id, role=node.role,
                level=ConformanceLevel.CONFORMING,
            )

        return ConformanceResult(
            node_id=node.node_id, role=node.role,
            level=ConformanceLevel.WARNING,
            warnings=(
                f"Interactive role '{node.role}' is not keyboard-focusable. "
                f"Consider adding tabindex='0' (§6.5)",
            ),
        )

    def check_landmarks(self, tree: AriaTree) -> List[ConformanceResult]:
        """Check landmark structure of the document.

        Requirements (WAI-ARIA 1.2 §5.3.4, WCAG 2.1 §1.3.1):
        - Page should have at most one ``banner`` landmark.
        - Page should have at most one ``contentinfo`` landmark.
        - Page should have exactly one ``main`` landmark.
        - Multiple landmarks of the same type should have distinct labels.

        Parameters:
            tree: The full :class:`AriaTree`.

        Returns:
            List of :class:`ConformanceResult` for landmark-level checks.
        """
        results: list[ConformanceResult] = []

        role_counts: Dict[str, list[AriaNodeInfo]] = {}
        for node in tree.node_index.values():
            if node.role in _LANDMARK_ROLES:
                role_counts.setdefault(node.role, []).append(node)

        # Check main landmark presence
        mains = role_counts.get("main", [])
        if len(mains) == 0:
            results.append(ConformanceResult(
                node_id="document", role="document",
                level=ConformanceLevel.WARNING,
                warnings=("Document has no 'main' landmark region (§5.3.4)",),
            ))
        elif len(mains) > 1:
            results.append(ConformanceResult(
                node_id="document", role="document",
                level=ConformanceLevel.WARNING,
                warnings=(
                    f"Document has {len(mains)} 'main' landmarks; "
                    f"at most one is recommended (§5.3.4)",
                ),
            ))

        # Check singular landmarks
        for singular in ("banner", "contentinfo"):
            nodes = role_counts.get(singular, [])
            if len(nodes) > 1:
                results.append(ConformanceResult(
                    node_id="document", role="document",
                    level=ConformanceLevel.WARNING,
                    warnings=(
                        f"Document has {len(nodes)} '{singular}' landmarks; "
                        f"at most one is recommended (§5.3.4)",
                    ),
                ))

        # Check duplicate landmarks have distinct labels
        for role_name, nodes in role_counts.items():
            if len(nodes) > 1:
                labels = [n.accessible_name for n in nodes]
                if len(set(labels)) < len(labels):
                    results.append(ConformanceResult(
                        node_id=nodes[0].node_id, role=role_name,
                        level=ConformanceLevel.WARNING,
                        warnings=(
                            f"Multiple '{role_name}' landmarks should have "
                            f"distinct accessible names (§5.3.4)",
                        ),
                    ))

        return results

    # ── Comprehensive check ───────────────────────────────────────────────

    def run_all_checks(self, tree: AriaTree) -> List[ConformanceResult]:
        """Run all conformance checks on an :class:`AriaTree`.

        Combines per-node checks (required properties, allowed properties,
        required children, required parent, accessible name, focusability)
        with document-level landmark checks.

        Parameters:
            tree: The :class:`AriaTree` to validate.

        Returns:
            Comprehensive list of :class:`ConformanceResult` instances.
        """
        results: list[ConformanceResult] = []

        for node in tree.node_index.values():
            # Skip abstract / unknown roles that have no spec constraints
            role_def = get_role(node.role)
            if role_def is not None and role_def.is_abstract:
                results.append(ConformanceResult(
                    node_id=node.node_id, role=node.role,
                    level=ConformanceLevel.NOT_APPLICABLE,
                    warnings=(
                        f"Abstract role '{node.role}' must not appear in "
                        f"authored content (§5.3.1)",
                    ),
                ))
                continue

            # Per-node checks
            results.append(self.check_required_properties(node))
            results.append(self.check_allowed_properties(node))
            results.append(self.check_required_children(node))
            results.append(self.check_required_parent(node, tree))
            results.append(self.check_name_required(node))
            results.append(self.check_focusable(node))

        # Document-level checks
        results.extend(self.check_landmarks(tree))

        return results

    # ── Internal helpers ──────────────────────────────────────────────────

    def _collect_child_roles(self, node: AriaNodeInfo) -> set[str]:
        """Collect all direct and accessible-owned child role names."""
        roles: set[str] = set()
        for child in node.children:
            roles.add(child.role)
        # Also check aria-owns references (if stored in properties)
        return roles

    def _collect_ancestor_roles(
        self, node: AriaNodeInfo, tree: AriaTree,
    ) -> set[str]:
        """Walk up the tree and collect all ancestor role names."""
        roles: set[str] = set()
        current_id = node.parent_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            parent = tree.get_node(current_id)
            if parent is None:
                break
            roles.add(parent.role)
            current_id = parent.parent_id
        return roles
