"""Accessibility tree validation with configurable rules.

Checks structural integrity, ARIA role validity, naming, parent-child
consistency, focusability, bounding-box containment, and tree depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.accessibility.roles import RoleTaxonomy


# ── Result types ──────────────────────────────────────────────────────────────

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: Severity
    message: str
    node_id: Optional[str] = None
    rule: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "node_id": self.node_id,
            "rule": self.rule,
        }


@dataclass
class ValidationResult:
    """Aggregate result of tree validation."""

    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "warnings": [w.to_dict() for w in self.warnings],
        }


# ── Validator class ───────────────────────────────────────────────────────────

_taxonomy = RoleTaxonomy()


class TreeValidator:
    """Validate an :class:`AccessibilityTree` against configurable rules."""

    def __init__(
        self,
        *,
        max_depth: int = 50,
        require_names_for_interactive: bool = True,
        check_containment: bool = True,
        containment_tolerance: float = 5.0,
    ) -> None:
        self.max_depth = max_depth
        self.require_names_for_interactive = require_names_for_interactive
        self.check_containment = check_containment
        self.containment_tolerance = containment_tolerance

    # ── Public API ────────────────────────────────────────────────────────

    def validate(self, tree: AccessibilityTree) -> ValidationResult:
        """Run all validation checks and return a :class:`ValidationResult`."""
        all_issues: list[ValidationIssue] = []

        all_issues.extend(self._check_unique_ids(tree))
        all_issues.extend(self._check_role_validity(tree))
        all_issues.extend(self._check_name_presence(tree))
        all_issues.extend(self._check_parent_child_consistency(tree))
        all_issues.extend(self._check_focusable_elements(tree))
        if self.check_containment:
            all_issues.extend(self._check_bounding_box_containment(tree))
        all_issues.extend(self._check_tree_depth(tree, self.max_depth))
        all_issues.extend(self._check_role_containment(tree))

        errors = [i for i in all_issues if i.severity == Severity.ERROR]
        warnings = [i for i in all_issues if i.severity in (Severity.WARNING, Severity.INFO)]

        return ValidationResult(
            valid=len(errors) == 0,
            issues=all_issues,
            warnings=warnings,
        )

    # ── Individual checks ─────────────────────────────────────────────────

    def _check_unique_ids(self, tree: AccessibilityTree) -> list[ValidationIssue]:
        """Every node must have a unique id."""
        issues: list[ValidationIssue] = []
        seen: dict[str, int] = {}
        for node in tree.root.iter_preorder():
            seen[node.id] = seen.get(node.id, 0) + 1

        for nid, count in seen.items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        message=f"Duplicate node id {nid!r} appears {count} times",
                        node_id=nid,
                        rule="unique-ids",
                    )
                )
        return issues

    def _check_role_validity(self, tree: AccessibilityTree) -> list[ValidationIssue]:
        """All roles should be recognised ARIA roles."""
        issues: list[ValidationIssue] = []
        known = _taxonomy.all_roles()
        # Also accept common non-ARIA roles that are used in practice
        extra_valid = {"text", "generic", "document", "window", "article", "figcaption"}

        for node in tree.node_index.values():
            if node.role and node.role not in known and node.role not in extra_valid:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        message=f"Unrecognised role {node.role!r}",
                        node_id=node.id,
                        rule="valid-role",
                    )
                )
        return issues

    def _check_name_presence(self, tree: AccessibilityTree) -> list[ValidationIssue]:
        """Interactive elements should have accessible names."""
        issues: list[ValidationIssue] = []
        if not self.require_names_for_interactive:
            return issues

        for node in tree.node_index.values():
            if node.is_interactive() and not node.name.strip():
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        message=f"Interactive node {node.id!r} (role={node.role!r}) has no accessible name",
                        node_id=node.id,
                        rule="name-presence",
                    )
                )

        # Images should have alt text (name)
        for node in tree.get_nodes_by_role("img"):
            if not node.name.strip() and node.role != "presentation":
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        message=f"Image node {node.id!r} has no accessible name (missing alt text)",
                        node_id=node.id,
                        rule="img-alt",
                    )
                )

        return issues

    def _check_parent_child_consistency(self, tree: AccessibilityTree) -> list[ValidationIssue]:
        """Verify parent_id links are consistent with children lists."""
        issues: list[ValidationIssue] = []

        for node in tree.node_index.values():
            for child in node.children:
                if child.parent_id != node.id:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            message=(
                                f"Child {child.id!r} of {node.id!r} has "
                                f"parent_id={child.parent_id!r} instead of {node.id!r}"
                            ),
                            node_id=child.id,
                            rule="parent-child-consistency",
                        )
                    )

                if child.id not in tree.node_index:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            message=f"Child {child.id!r} of {node.id!r} is not in the node index",
                            node_id=child.id,
                            rule="index-completeness",
                        )
                    )

        # Check depth consistency
        for node in tree.node_index.values():
            for child in node.children:
                if child.depth != node.depth + 1:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            message=(
                                f"Child {child.id!r} depth={child.depth} does not equal "
                                f"parent {node.id!r} depth={node.depth} + 1"
                            ),
                            node_id=child.id,
                            rule="depth-consistency",
                        )
                    )

        return issues

    def _check_focusable_elements(self, tree: AccessibilityTree) -> list[ValidationIssue]:
        """Warn if there are no focusable elements, or disabled elements claim focus."""
        issues: list[ValidationIssue] = []

        focusable = tree.get_focusable_nodes()
        interactive = tree.get_interactive_nodes()

        if interactive and not focusable:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    message=(
                        f"Tree has {len(interactive)} interactive nodes but none are focusable"
                    ),
                    rule="focusable-exists",
                )
            )

        for node in tree.node_index.values():
            if node.state.focused and node.state.disabled:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        message=f"Node {node.id!r} is both focused and disabled",
                        node_id=node.id,
                        rule="focus-disabled-conflict",
                    )
                )

            if node.state.focused and node.state.hidden:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        message=f"Node {node.id!r} is both focused and hidden",
                        node_id=node.id,
                        rule="focus-hidden-conflict",
                    )
                )

        return issues

    def _check_bounding_box_containment(
        self, tree: AccessibilityTree,
    ) -> list[ValidationIssue]:
        """Check that child bounding boxes are contained within parent boxes."""
        issues: list[ValidationIssue] = []
        tol = self.containment_tolerance

        for node in tree.node_index.values():
            if node.bounding_box is None:
                continue
            pb = node.bounding_box
            for child in node.children:
                if child.bounding_box is None:
                    continue
                cb = child.bounding_box
                # Allow tolerance pixels of overflow
                if (
                    cb.x < pb.x - tol
                    or cb.y < pb.y - tol
                    or cb.right > pb.right + tol
                    or cb.bottom > pb.bottom + tol
                ):
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            message=(
                                f"Child {child.id!r} bounding box exceeds parent "
                                f"{node.id!r} by more than {tol}px"
                            ),
                            node_id=child.id,
                            rule="bbox-containment",
                        )
                    )
        return issues

    def _check_tree_depth(
        self, tree: AccessibilityTree, max_depth: int,
    ) -> list[ValidationIssue]:
        """Warn if tree exceeds the maximum depth threshold."""
        issues: list[ValidationIssue] = []
        actual_depth = tree.depth()
        if actual_depth > max_depth:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    message=f"Tree depth {actual_depth} exceeds maximum {max_depth}",
                    rule="max-depth",
                )
            )

        # Also flag individual very deep nodes
        for node in tree.node_index.values():
            if node.depth > max_depth:
                issues.append(
                    ValidationIssue(
                        severity=Severity.INFO,
                        message=f"Node {node.id!r} at depth {node.depth} exceeds max {max_depth}",
                        node_id=node.id,
                        rule="node-max-depth",
                    )
                )
                break  # only one report needed

        return issues

    def _check_role_containment(self, tree: AccessibilityTree) -> list[ValidationIssue]:
        """Check ARIA role containment rules (e.g., list should contain listitem)."""
        issues: list[ValidationIssue] = []

        for node in tree.node_index.values():
            expected = _taxonomy.expected_children(node.role)
            if not expected or not node.children:
                continue

            for child in node.children:
                if child.role in ("text", "generic", "none", "presentation"):
                    continue
                if not _taxonomy.can_contain(node.role, child.role):
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            message=(
                                f"Node {node.id!r} (role={node.role!r}) contains "
                                f"child {child.id!r} (role={child.role!r}) which is not "
                                f"in expected set {expected}"
                            ),
                            node_id=child.id,
                            rule="role-containment",
                        )
                    )
        return issues
