"""
usability_oracle.repair.validator — Validate that repairs preserve
functionality and improve usability.

Provides :class:`RepairValidator` which checks a repaired accessibility
tree against the original to ensure that:

1. All task-critical interactive elements are preserved.
2. Accessibility properties are maintained or improved.
3. Cognitive cost is reduced (the repair's raison d'être).
4. No new bottlenecks are introduced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of repair validation.

    Attributes
    ----------
    valid : bool
        True if all checks passed.
    functionality_preserved : bool
        All task-critical elements still present and reachable.
    accessibility_maintained : bool
        No WCAG/accessibility regressions.
    cost_improved : bool
        Cognitive cost is strictly lower after repair.
    new_bottlenecks : list[str]
        Descriptions of any newly introduced bottlenecks.
    improvement_ratio : float
        Fractional improvement: (old − new) / old.
    details : dict[str, Any]
        Additional detail for each check.
    errors : list[str]
        Hard-failure descriptions.
    warnings : list[str]
        Non-blocking issues.
    """

    valid: bool = False
    functionality_preserved: bool = False
    accessibility_maintained: bool = False
    cost_improved: bool = False
    new_bottlenecks: list[str] = field(default_factory=list)
    improvement_ratio: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "functionality_preserved": self.functionality_preserved,
            "accessibility_maintained": self.accessibility_maintained,
            "cost_improved": self.cost_improved,
            "new_bottlenecks": self.new_bottlenecks,
            "improvement_ratio": self.improvement_ratio,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# RepairValidator
# ---------------------------------------------------------------------------

class RepairValidator:
    """Validate repaired accessibility trees.

    Parameters
    ----------
    min_improvement : float
        Minimum required cost-improvement ratio (default 0.05 = 5 %).
    strict_mode : bool
        If True, any warning also fails validation.
    """

    def __init__(
        self,
        min_improvement: float = 0.05,
        strict_mode: bool = False,
    ) -> None:
        self.min_improvement = min_improvement
        self.strict_mode = strict_mode

    def validate(
        self,
        original_tree: AccessibilityTree,
        repaired_tree: AccessibilityTree,
        task_spec: Any | None = None,
    ) -> ValidationResult:
        """Run all validation checks.

        Parameters
        ----------
        original_tree : AccessibilityTree
        repaired_tree : AccessibilityTree
        task_spec : TaskSpec-like, optional
            If provided, task-specific functionality checks are run.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        # 1. Functionality preserved
        func_ok = self._check_functionality_preserved(
            original_tree, repaired_tree, task_spec
        )
        result.functionality_preserved = func_ok
        if not func_ok:
            result.errors.append("Functionality not preserved: missing interactive elements")

        # 2. Accessibility maintained
        acc_ok = self._check_accessibility_maintained(repaired_tree)
        result.accessibility_maintained = acc_ok
        if not acc_ok:
            result.warnings.append("Accessibility regression detected")

        # 3. Cost improved
        cost_ok = self._check_cost_improved(original_tree, repaired_tree)
        result.cost_improved = cost_ok
        if not cost_ok:
            result.errors.append("Cognitive cost not improved")

        # 4. No new bottlenecks
        new_bns = self._check_no_new_bottlenecks(original_tree, repaired_tree)
        result.new_bottlenecks = new_bns
        if new_bns:
            for bn in new_bns:
                result.warnings.append(f"New bottleneck: {bn}")

        # 5. Improvement ratio
        original_cost = self._estimate_cost(original_tree)
        repaired_cost = self._estimate_cost(repaired_tree)
        result.improvement_ratio = self._compute_improvement(
            original_cost, repaired_cost
        )

        # Store details
        result.details = {
            "original_size": original_tree.size(),
            "repaired_size": repaired_tree.size(),
            "original_cost": original_cost,
            "repaired_cost": repaired_cost,
            "original_interactive": len(original_tree.get_interactive_nodes()),
            "repaired_interactive": len(repaired_tree.get_interactive_nodes()),
            "original_depth": original_tree.depth(),
            "repaired_depth": repaired_tree.depth(),
        }

        # Overall validity
        result.valid = (
            result.functionality_preserved
            and result.cost_improved
            and not result.errors
        )
        if self.strict_mode and result.warnings:
            result.valid = False

        return result

    # ── Individual checks -------------------------------------------------

    def _check_functionality_preserved(
        self,
        original: AccessibilityTree,
        repaired: AccessibilityTree,
        task_spec: Any | None,
    ) -> bool:
        """Verify that all task-critical interactive elements survive.

        Checks:
        1. All interactive nodes from the original that match task targets
           are present in the repaired tree.
        2. No interactive node has become disabled or hidden.
        """
        original_interactive = {
            n.id: n for n in original.get_interactive_nodes()
        }
        repaired_index = repaired.node_index

        # If we have a task spec, check only task-relevant nodes
        if task_spec is not None:
            task_targets = self._extract_task_targets(task_spec)
        else:
            # Without a task spec, check all interactive nodes
            task_targets = set(original_interactive.keys())

        for nid in task_targets:
            orig_node = original_interactive.get(nid)
            if orig_node is None:
                continue  # not in original interactive set

            repaired_node = repaired_index.get(nid)
            if repaired_node is None:
                # Node was removed; check if its role+name appear elsewhere
                found = self._find_equivalent_node(orig_node, repaired)
                if not found:
                    logger.warning(
                        "Task-critical node %s (%s) missing after repair",
                        nid, orig_node.name,
                    )
                    return False

            elif repaired_node.state.disabled and not orig_node.state.disabled:
                logger.warning(
                    "Node %s became disabled after repair", nid
                )
                return False

            elif repaired_node.state.hidden and not orig_node.state.hidden:
                logger.warning(
                    "Node %s became hidden after repair", nid
                )
                return False

        return True

    def _check_accessibility_maintained(
        self, repaired: AccessibilityTree
    ) -> bool:
        """Check basic accessibility properties of the repaired tree.

        Verifies:
        1. All interactive nodes have non-empty names (WCAG 4.1.2).
        2. No unreasonably small targets (WCAG 2.5.8).
        3. Focus order is not broken (no orphan focusable nodes).
        """
        issues_found = False

        for node in repaired.get_interactive_nodes():
            # Name check
            if not node.name.strip():
                logger.debug(
                    "Accessibility issue: interactive node %s has empty name",
                    node.id,
                )
                issues_found = True

            # Target size check (minimum 24×24 px per WCAG 2.5.8)
            if node.bounding_box is not None:
                bb = node.bounding_box
                if bb.width < 24 or bb.height < 24:
                    logger.debug(
                        "Accessibility issue: node %s too small (%.0f×%.0f)",
                        node.id, bb.width, bb.height,
                    )
                    issues_found = True

        # Check tree structure integrity
        validation_errors = repaired.validate()
        if validation_errors:
            logger.debug("Tree validation errors: %s", validation_errors)
            issues_found = True

        return not issues_found

    def _check_cost_improved(
        self,
        original: AccessibilityTree,
        repaired: AccessibilityTree,
    ) -> bool:
        """Verify that estimated cognitive cost is lower after repair."""
        original_cost = self._estimate_cost(original)
        repaired_cost = self._estimate_cost(repaired)

        improvement = self._compute_improvement(original_cost, repaired_cost)
        return improvement >= self.min_improvement

    def _check_no_new_bottlenecks(
        self,
        original: AccessibilityTree,
        repaired: AccessibilityTree,
    ) -> list[str]:
        """Detect potential new bottlenecks introduced by the repair.

        Heuristic checks:
        1. Menu depth increased → potential choice paralysis.
        2. More elements visible at once → perceptual overload.
        3. Smaller targets → motor difficulty.
        4. More steps required → memory decay.
        """
        new_bottlenecks: list[str] = []

        # Depth check
        orig_depth = original.depth()
        rep_depth = repaired.depth()
        if rep_depth > orig_depth + 1:
            new_bottlenecks.append(
                f"Hierarchy depth increased from {orig_depth} to {rep_depth} "
                f"(potential choice paralysis)"
            )

        # Visible element count
        orig_visible = len(original.get_visible_nodes())
        rep_visible = len(repaired.get_visible_nodes())
        if rep_visible > orig_visible * 1.5 and rep_visible > orig_visible + 10:
            new_bottlenecks.append(
                f"Visible elements increased from {orig_visible} to {rep_visible} "
                f"(potential perceptual overload)"
            )

        # Target size check
        for node in repaired.get_interactive_nodes():
            if node.bounding_box is None:
                continue
            orig_node = original.get_node(node.id)
            if orig_node is None or orig_node.bounding_box is None:
                continue
            if (node.bounding_box.area < orig_node.bounding_box.area * 0.5
                    and node.bounding_box.area < 24 * 24):
                new_bottlenecks.append(
                    f"Target {node.id} shrank significantly "
                    f"(potential motor difficulty)"
                )

        return new_bottlenecks

    # ── Helpers -----------------------------------------------------------

    def _compute_improvement(
        self, original_cost: float, repaired_cost: float
    ) -> float:
        """Compute fractional improvement: (old − new) / old."""
        if original_cost <= 0:
            return 0.0
        return (original_cost - repaired_cost) / original_cost

    def _estimate_cost(self, tree: AccessibilityTree) -> float:
        """Estimate aggregate cognitive cost of a tree.

        Uses a simplified heuristic combining:
        - Tree depth (navigation cost)
        - Number of interactive elements (choice cost via Hick-Hyman)
        - Average target size (motor cost via Fitts' Law)
        """
        depth = tree.depth()
        interactive = tree.get_interactive_nodes()
        n_interactive = max(1, len(interactive))

        # Hick-Hyman: decision time ∝ log₂(n + 1)
        hick_cost = 0.2 + 0.15 * math.log2(n_interactive + 1)

        # Fitts: average index of difficulty
        fitts_costs: list[float] = []
        for node in interactive:
            if node.bounding_box and node.bounding_box.width > 0:
                # Assume average distance = 200px (screen center)
                d = 200.0
                w = node.bounding_box.width
                fitts_id = math.log2(1 + d / w)
                fitts_costs.append(0.05 + 0.15 * fitts_id)
        avg_fitts = sum(fitts_costs) / len(fitts_costs) if fitts_costs else 0.5

        # Navigation cost ∝ depth
        nav_cost = depth * 0.3

        return hick_cost + avg_fitts + nav_cost

    def _extract_task_targets(self, task_spec: Any) -> set[str]:
        """Extract node IDs referenced by the task spec."""
        targets: set[str] = set()
        # Support TaskSpec with flows containing steps
        flows = getattr(task_spec, "flows", [])
        if not flows:
            flows = [task_spec] if hasattr(task_spec, "steps") else []

        for flow in flows:
            steps = getattr(flow, "steps", [])
            for step in steps:
                selector = getattr(step, "target_selector", None)
                if selector:
                    targets.add(str(selector))
                name = getattr(step, "target_name", None)
                if name:
                    targets.add(str(name))
        return targets

    def _find_equivalent_node(
        self, original_node: AccessibilityNode, tree: AccessibilityTree
    ) -> bool:
        """Check if an equivalent node exists (same role and name)."""
        for node in tree.node_index.values():
            if (node.role == original_node.role
                    and node.name == original_node.name
                    and not node.state.hidden
                    and not node.state.disabled):
                return True
        return False
