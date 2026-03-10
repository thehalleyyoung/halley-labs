"""
usability_oracle.smt_repair.validator — Repair validation.

Validates that proposed mutations actually improve usability without
introducing regressions.  The validation pipeline checks:

1. **Structural validity** — the repaired tree is well-formed.
2. **Accessibility compliance** — ARIA roles remain valid.
3. **Task completability** — all task flows are still achievable.
4. **Cost improvement** — cognitive cost is actually reduced.
5. **Side-effect check** — no regressions on unrelated tasks.

Implements the :class:`~usability_oracle.smt_repair.protocols.MutationValidator`
protocol.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.smt_repair.types import (
    MutationCandidate,
    MutationType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARIA role tables for compliance checking
# ---------------------------------------------------------------------------

_VALID_CHILD_ROLES: Dict[str, frozenset] = {
    "list": frozenset({"listitem", "group"}),
    "menu": frozenset({"menuitem", "menuitemcheckbox", "menuitemradio", "group", "separator"}),
    "menubar": frozenset({"menuitem", "menuitemcheckbox", "menuitemradio", "menu", "separator"}),
    "tablist": frozenset({"tab"}),
    "radiogroup": frozenset({"radio"}),
    "tree": frozenset({"treeitem", "group"}),
    "grid": frozenset({"row", "rowgroup"}),
    "row": frozenset({"cell", "gridcell", "columnheader", "rowheader"}),
    "table": frozenset({"row", "rowgroup"}),
    "rowgroup": frozenset({"row"}),
    "listbox": frozenset({"option", "group"}),
}

_INTERACTIVE_ROLES: frozenset = frozenset({
    "button", "checkbox", "combobox", "link", "listbox", "menu",
    "menuitem", "menuitemcheckbox", "menuitemradio", "option",
    "radio", "scrollbar", "slider", "spinbutton", "switch",
    "tab", "textbox", "treeitem",
})


# ═══════════════════════════════════════════════════════════════════════════
# RepairValidator
# ═══════════════════════════════════════════════════════════════════════════

class RepairValidator:
    """Validates proposed UI repairs.

    Implements the :class:`~usability_oracle.smt_repair.protocols.MutationValidator`
    protocol.
    """

    # ── protocol methods ──────────────────────────────────────────────

    def validate(
        self,
        mutations: Sequence[MutationCandidate],
        original_tree: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate mutations against the original tree.

        Parameters:
            mutations: Proposed mutations from the SMT solver.
            original_tree: Serialised accessibility tree before repair.

        Returns:
            Validation report with ``"valid"``, ``"violations"``, and
            ``"estimated_improvement"`` fields.
        """
        repaired = self.apply_mutations(mutations, original_tree)
        violations: List[Dict[str, Any]] = []

        # Structural validity.
        struct_issues = self.check_structural_validity(repaired)
        violations.extend(struct_issues)

        # Accessibility compliance.
        a11y_issues = self.check_accessibility_compliance(repaired)
        violations.extend(a11y_issues)

        # Cost improvement estimate.
        improvement = self.check_cost_improvement(original_tree, repaired)

        valid = len([v for v in violations if v.get("severity") == "error"]) == 0

        return {
            "valid": valid,
            "violations": violations,
            "estimated_improvement": improvement,
            "num_mutations": len(mutations),
        }

    def apply_mutations(
        self,
        mutations: Sequence[MutationCandidate],
        tree_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply mutations to a serialised tree and return the result.

        Parameters:
            mutations: Ordered sequence of mutations to apply.
            tree_dict: Serialised accessibility tree.

        Returns:
            Updated tree with mutations applied.
        """
        tree = copy.deepcopy(tree_dict)
        for mutation in mutations:
            tree = self._apply_single_mutation(tree, mutation)
        return tree

    # ── validation pipeline ───────────────────────────────────────────

    def validate_repair(
        self,
        original: Dict[str, Any],
        repaired: Dict[str, Any],
        task_specs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Full validation pipeline.

        Parameters:
            original: Original tree.
            repaired: Repaired tree.
            task_specs: Optional list of task specifications.

        Returns:
            Comprehensive validation report.
        """
        violations: List[Dict[str, Any]] = []

        violations.extend(self.check_structural_validity(repaired))
        violations.extend(self.check_accessibility_compliance(repaired))

        if task_specs:
            for task in task_specs:
                violations.extend(self.check_task_completability(repaired, task))

        improvement = self.check_cost_improvement(original, repaired)

        if task_specs:
            violations.extend(self.check_side_effects(original, repaired, task_specs))

        return self.generate_validation_report({
            "violations": violations,
            "improvement": improvement,
            "task_count": len(task_specs) if task_specs else 0,
        })

    def check_structural_validity(
        self,
        tree: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check that the tree is well-formed.

        Verifies:
        - Every node has an ``"id"`` field.
        - No duplicate IDs.
        - ``parent_id`` references are consistent.

        Returns:
            List of violation dicts (empty if valid).
        """
        violations: List[Dict[str, Any]] = []
        seen_ids: set = set()

        def _walk(node: Dict[str, Any], expected_parent: Optional[str]) -> None:
            nid = node.get("id")
            if nid is None:
                violations.append({
                    "type": "structural",
                    "severity": "error",
                    "message": "Node missing 'id' field",
                    "node": str(node.get("role", "unknown")),
                })
                return

            if nid in seen_ids:
                violations.append({
                    "type": "structural",
                    "severity": "error",
                    "message": f"Duplicate node ID: {nid}",
                    "node": nid,
                })
            seen_ids.add(nid)

            actual_parent = node.get("parent_id")
            if expected_parent is not None and actual_parent is not None:
                if str(actual_parent) != str(expected_parent):
                    violations.append({
                        "type": "structural",
                        "severity": "warning",
                        "message": f"Node {nid} parent_id mismatch: "
                                   f"expected {expected_parent}, got {actual_parent}",
                        "node": nid,
                    })

            for child in node.get("children", []):
                _walk(child, str(nid))

        _walk(tree, None)
        return violations

    def check_accessibility_compliance(
        self,
        tree: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check ARIA compliance: valid role nesting, labels, etc.

        Verifies:
        - Container roles have only allowed child roles.
        - Interactive elements have accessible names (non-empty ``name``).

        Returns:
            List of violation dicts.
        """
        violations: List[Dict[str, Any]] = []

        def _walk(node: Dict[str, Any]) -> None:
            nid = str(node.get("id", ""))
            role = str(node.get("role", "generic"))
            children = node.get("children", [])

            # Check child-role validity.
            valid_children = _VALID_CHILD_ROLES.get(role)
            if valid_children and children:
                for child in children:
                    child_role = str(child.get("role", "generic"))
                    if child_role not in valid_children and child_role != "generic":
                        violations.append({
                            "type": "accessibility",
                            "severity": "warning",
                            "message": f"Role '{child_role}' is not a valid child of '{role}' "
                                       f"(parent: {nid})",
                            "node": str(child.get("id", "")),
                        })

            # Interactive elements should have names.
            if role in _INTERACTIVE_ROLES:
                name = str(node.get("name", "")).strip()
                if not name:
                    violations.append({
                        "type": "accessibility",
                        "severity": "warning",
                        "message": f"Interactive element {nid} ({role}) has no accessible name",
                        "node": nid,
                    })

            for child in children:
                _walk(child)

        _walk(tree)
        return violations

    def check_task_completability(
        self,
        tree: Dict[str, Any],
        task: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check that all task flows are still achievable in the tree.

        For each step in every flow, verifies that a matching node
        (by role and optionally name) exists and is not hidden.

        Parameters:
            tree: Repaired accessibility tree.
            task: Task specification dict.

        Returns:
            List of violation dicts.
        """
        violations: List[Dict[str, Any]] = []
        nodes_by_role: Dict[str, List[Dict[str, Any]]] = {}
        self._index_by_role(tree, nodes_by_role)

        for flow in task.get("flows", []):
            flow_id = flow.get("flow_id", flow.get("name", ""))
            for step in flow.get("steps", []):
                target_role = step.get("target_role", "")
                target_name = step.get("target_name", "")
                optional = step.get("optional", False)

                if not target_role:
                    continue

                candidates = nodes_by_role.get(target_role, [])
                if target_name:
                    candidates = [
                        n for n in candidates
                        if target_name.lower() in str(n.get("name", "")).lower()
                    ]

                # Filter hidden.
                visible = [
                    n for n in candidates
                    if not n.get("state", {}).get("hidden", False)
                ]

                if not visible and not optional:
                    violations.append({
                        "type": "task",
                        "severity": "error",
                        "message": (
                            f"Task step requires {target_role}"
                            f"{' named ' + repr(target_name) if target_name else ''}"
                            f" but none found (flow: {flow_id})"
                        ),
                        "node": target_role,
                    })

        return violations

    def check_cost_improvement(
        self,
        original: Dict[str, Any],
        repaired: Dict[str, Any],
        task: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Estimate whether the repair reduces cognitive cost.

        Uses a heuristic depth-weighted score: shallower interactive
        elements and larger target sizes are better.

        Parameters:
            original: Original tree.
            repaired: Repaired tree.

        Returns:
            Estimated cost delta (negative = improvement).
        """
        original_score = self._compute_heuristic_cost(original)
        repaired_score = self._compute_heuristic_cost(repaired)
        return repaired_score - original_score

    def check_side_effects(
        self,
        original: Dict[str, Any],
        repaired: Dict[str, Any],
        all_tasks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for regressions on tasks not targeted by the repair.

        Runs :meth:`check_task_completability` for every task and
        reports any new failures.

        Parameters:
            original: Original tree.
            repaired: Repaired tree.
            all_tasks: All task specifications.

        Returns:
            List of regression violation dicts.
        """
        violations: List[Dict[str, Any]] = []
        for task in all_tasks:
            original_issues = self.check_task_completability(original, task)
            repaired_issues = self.check_task_completability(repaired, task)

            # New errors that didn't exist before are regressions.
            original_msgs = {v["message"] for v in original_issues}
            for v in repaired_issues:
                if v["message"] not in original_msgs and v["severity"] == "error":
                    v["type"] = "regression"
                    violations.append(v)

        return violations

    @staticmethod
    def generate_validation_report(results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured validation report.

        Parameters:
            results: Intermediate results with ``"violations"``,
                ``"improvement"``, and ``"task_count"`` keys.

        Returns:
            Structured report dict.
        """
        violations = results.get("violations", [])
        errors = [v for v in violations if v.get("severity") == "error"]
        warnings = [v for v in violations if v.get("severity") == "warning"]

        return {
            "valid": len(errors) == 0,
            "violations": violations,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "estimated_improvement": results.get("improvement", 0.0),
            "task_count": results.get("task_count", 0),
            "summary": (
                "Repair is valid" if len(errors) == 0
                else f"Repair has {len(errors)} error(s)"
            ),
        }

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _apply_single_mutation(
        tree: Dict[str, Any],
        mutation: MutationCandidate,
    ) -> Dict[str, Any]:
        """Apply a single MutationCandidate to a tree dict."""
        node = _find_node(tree, mutation.node_id)
        if node is None:
            return tree

        if mutation.mutation_type == MutationType.PROPERTY_CHANGE:
            prop = mutation.property_name
            if prop is None:
                return tree

            # Map property to the correct location in the node dict.
            if prop in ("x", "y", "width", "height"):
                bbox = node.get("bounding_box", {})
                bbox[prop] = mutation.new_value
                node["bounding_box"] = bbox
            elif prop in ("hidden", "focused", "disabled", "selected"):
                state = node.get("state", {})
                state[prop] = mutation.new_value
                node["state"] = state
            elif prop == "role":
                node["role"] = mutation.new_value
            elif prop == "name":
                node["name"] = mutation.new_value
            elif prop == "name_len":
                pass  # Derived property, skip.
            else:
                props = node.get("properties", {})
                props[prop] = mutation.new_value
                node["properties"] = props

        elif mutation.mutation_type == MutationType.ELEMENT_REMOVE:
            _remove_node_from_tree(tree, mutation.node_id)

        return tree

    @staticmethod
    def _index_by_role(
        tree: Dict[str, Any],
        result: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Build an index from role to list of nodes."""
        role = str(tree.get("role", "generic"))
        result.setdefault(role, []).append(tree)
        for child in tree.get("children", []):
            RepairValidator._index_by_role(child, result)

    @staticmethod
    def _compute_heuristic_cost(tree: Dict[str, Any]) -> float:
        """Compute a heuristic cognitive cost for the tree.

        Cost increases with depth of interactive elements and decreases
        with target size.
        """
        total = 0.0

        def _walk(node: Dict[str, Any], depth: int) -> None:
            nonlocal total
            role = str(node.get("role", "generic"))
            if role in _INTERACTIVE_ROLES:
                bbox = node.get("bounding_box", {})
                w = max(bbox.get("width", 1), 1)
                h = max(bbox.get("height", 1), 1)
                # Smaller targets and deeper elements cost more.
                total += depth * (1.0 + 100.0 / (w * h))
            for child in node.get("children", []):
                _walk(child, depth + 1)

        _walk(tree, 0)
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Tree helpers
# ═══════════════════════════════════════════════════════════════════════════

def _find_node(tree: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """Find a node by ID (DFS)."""
    if str(tree.get("id", "")) == node_id:
        return tree
    for child in tree.get("children", []):
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def _remove_node_from_tree(tree: Dict[str, Any], node_id: str) -> bool:
    """Remove a node by ID from the tree."""
    children = tree.get("children", [])
    for i, child in enumerate(children):
        if str(child.get("id", "")) == node_id:
            children.pop(i)
            return True
        if _remove_node_from_tree(child, node_id):
            return True
    return False
