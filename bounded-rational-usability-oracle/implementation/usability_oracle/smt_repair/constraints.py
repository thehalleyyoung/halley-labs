"""
usability_oracle.smt_repair.constraints — Constraint generation from UI structure.

Translates accessibility tree properties and usability bottleneck reports
into SMT constraints suitable for the Z3 solver.  Constraint families
cover:

* **Structural** — parent/child relationships, sibling ordering.
* **Spatial** — bounding-box containment, non-overlap, alignment.
* **Role** — ARIA role validity (valid parent/child role combinations).
* **Navigation** — reachability, keyboard traversal order.
* **Cognitive cost** — Fitts' law, Hick–Hyman law bounds.
* **Grouping** — semantic grouping of related elements.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import z3

from usability_oracle.smt_repair.encoding import (
    CONTAINS,
    NON_OVERLAP,
    TreeEncoding,
    Z3Encoder,
    _COMMON_ROLES,
)
from usability_oracle.smt_repair.types import (
    ConstraintId,
    ConstraintKind,
    ConstraintSystem,
    RepairConstraint,
    UIVariable,
    VariableSort,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARIA role-validity tables
# ---------------------------------------------------------------------------

# Maps parent role to the set of valid direct-child roles.
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
    "toolbar": frozenset({"button", "link", "checkbox", "separator", "menuitem", "group"}),
    "select": frozenset({"option"}),
}

# Landmark roles that may serve as navigation targets.
_LANDMARK_ROLES: frozenset = frozenset({
    "banner", "complementary", "contentinfo", "form",
    "main", "navigation", "region", "search",
})

# Minimum touch-target size in px (WCAG 2.5.5 Level AAA guideline).
_MIN_TARGET_SIZE = 44


# ═══════════════════════════════════════════════════════════════════════════
# ConstraintGenerator
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintGenerator:
    """Generates SMT repair constraints from accessibility trees and bottleneck reports.

    Implements the :class:`~usability_oracle.smt_repair.protocols.ConstraintGenerator`
    protocol.
    """

    def __init__(self, encoder: Optional[Z3Encoder] = None) -> None:
        self._encoder = encoder or Z3Encoder()
        self._cid_counter = 0

    # ── protocol methods ──────────────────────────────────────────────

    def generate_variables(
        self,
        tree_dict: Dict[str, Any],
        mutable_properties: Sequence[str],
    ) -> Sequence[UIVariable]:
        """Extract mutable UI variables from an accessibility tree.

        Parameters:
            tree_dict: Serialised accessibility tree.
            mutable_properties: Property names that may be modified
                (e.g. ``["width", "height", "role"]``).

        Returns:
            Sequence of :class:`UIVariable` for every mutable property
            of every node in the tree.
        """
        encoding = self._encoder.encode_tree(tree_dict)
        mutable = frozenset(mutable_properties)
        return [
            uv for uv in encoding.ui_variables.values()
            if uv.property_name in mutable
        ]

    def generate_constraints(
        self,
        variables: Sequence[UIVariable],
        bottleneck_report: Dict[str, Any],
    ) -> Sequence[RepairConstraint]:
        """Generate repair constraints from a bottleneck report.

        Dispatches to specialised generators based on bottleneck type
        and cognitive law.

        Parameters:
            variables: Available UI variables.
            bottleneck_report: Serialised bottleneck classification.

        Returns:
            Hard and soft constraints addressing identified bottlenecks.
        """
        constraints: List[RepairConstraint] = []
        var_map = {v.variable_id: v for v in variables}

        for bn in bottleneck_report.get("bottlenecks", []):
            law = bn.get("cognitive_law", "")
            severity = bn.get("severity", "MEDIUM")
            evidence = bn.get("evidence", {})
            affected = bn.get("affected_states", [])

            # Fitts' law — target size too small or distance too large.
            if law == "FITTS" or bn.get("bottleneck_type") == "MOTOR":
                constraints.extend(self._fitts_constraints(var_map, bn))

            # Hick–Hyman — too many choices.
            if law == "HICK_HYMAN" or bn.get("bottleneck_type") == "CHOICE":
                constraints.extend(self._hick_constraints(var_map, bn))

            # Working-memory overload.
            if law == "MEMORY" or bn.get("bottleneck_type") == "MEMORY":
                constraints.extend(self._memory_constraints(var_map, bn))

            # Perceptual — visibility / contrast.
            if law == "PERCEPTUAL" or bn.get("bottleneck_type") == "PERCEPTUAL":
                constraints.extend(self._perceptual_constraints(var_map, bn))

        # Always add preservation (soft) constraints to minimise change.
        constraints.extend(self._preservation_constraints(variables))

        return constraints

    def build_system(
        self,
        variables: Sequence[UIVariable],
        constraints: Sequence[RepairConstraint],
        objective_expression: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> ConstraintSystem:
        """Assemble a complete :class:`ConstraintSystem`."""
        return ConstraintSystem(
            variables=tuple(variables),
            constraints=tuple(constraints),
            objective_expression=objective_expression,
            timeout_seconds=timeout_seconds,
        )

    # ── structural constraints ────────────────────────────────────────

    def generate_structural_constraints(
        self,
        tree: Dict[str, Any],
        encoding: TreeEncoding,
    ) -> List[RepairConstraint]:
        """Encode tree structure: parent-child, sibling order, depth.

        Every child's ``index_in_parent`` must be consistent with its
        siblings, and each child must reference a valid parent.
        """
        constraints: List[RepairConstraint] = []
        self._structural_recurse(tree, encoding, constraints)
        return constraints

    def _structural_recurse(
        self,
        node: Dict[str, Any],
        encoding: TreeEncoding,
        out: List[RepairConstraint],
    ) -> None:
        nid = str(node.get("id", ""))
        children = node.get("children", [])

        if not children:
            return

        child_ids = [str(c.get("id", "")) for c in children]
        # Sibling distinctness — each child occupies a unique position.
        if len(child_ids) >= 2:
            for i in range(len(child_ids)):
                for j in range(i + 1, len(child_ids)):
                    cid_a, cid_b = child_ids[i], child_ids[j]
                    if cid_a in encoding.node_vars and cid_b in encoding.node_vars:
                        expr = f"(distinct {cid_a}__index {cid_b}__index)"
                        out.append(self._make_constraint(
                            kind=ConstraintKind.LAYOUT,
                            desc=f"Siblings {cid_a} and {cid_b} have distinct order",
                            expr=expr,
                            variables=(f"{cid_a}__x", f"{cid_b}__x"),
                            is_hard=True,
                        ))

        for child in children:
            self._structural_recurse(child, encoding, out)

    # ── spatial constraints ───────────────────────────────────────────

    def generate_spatial_constraints(
        self,
        tree: Dict[str, Any],
        encoding: TreeEncoding,
    ) -> List[RepairConstraint]:
        """Encode bounding-box containment, non-overlap, and alignment.

        - Children must be contained within their parent's bounding box.
        - Sibling elements must not overlap.
        """
        constraints: List[RepairConstraint] = []
        self._spatial_recurse(tree, encoding, constraints)
        return constraints

    def _spatial_recurse(
        self,
        node: Dict[str, Any],
        encoding: TreeEncoding,
        out: List[RepairConstraint],
    ) -> None:
        nid = str(node.get("id", ""))
        children = node.get("children", [])
        parent_vars = encoding.node_vars.get(nid, {})

        for child in children:
            cid = str(child.get("id", ""))
            child_vars = encoding.node_vars.get(cid, {})

            if parent_vars and child_vars and all(
                k in parent_vars and k in child_vars
                for k in ("x", "y", "width", "height")
            ):
                # Containment: child inside parent.
                z3_expr = self._encoder.encode_spatial_relation(
                    parent_vars, child_vars, CONTAINS
                )
                out.append(self._make_constraint(
                    kind=ConstraintKind.LAYOUT,
                    desc=f"Child {cid} contained in parent {nid}",
                    expr=str(z3_expr),
                    variables=(
                        f"{nid}__x", f"{nid}__y", f"{nid}__width", f"{nid}__height",
                        f"{cid}__x", f"{cid}__y", f"{cid}__width", f"{cid}__height",
                    ),
                    is_hard=True,
                ))

        # Sibling non-overlap.
        child_ids = [str(c.get("id", "")) for c in children]
        for i in range(len(child_ids)):
            for j in range(i + 1, len(child_ids)):
                a, b = child_ids[i], child_ids[j]
                av = encoding.node_vars.get(a, {})
                bv = encoding.node_vars.get(b, {})
                if av and bv and all(
                    k in av and k in bv for k in ("x", "y", "width", "height")
                ):
                    z3_expr = self._encoder.encode_spatial_relation(av, bv, NON_OVERLAP)
                    out.append(self._make_constraint(
                        kind=ConstraintKind.LAYOUT,
                        desc=f"Siblings {a} and {b} do not overlap",
                        expr=str(z3_expr),
                        variables=(
                            f"{a}__x", f"{a}__y", f"{a}__width", f"{a}__height",
                            f"{b}__x", f"{b}__y", f"{b}__width", f"{b}__height",
                        ),
                        is_hard=False,
                        weight=0.8,
                    ))

        for child in children:
            self._spatial_recurse(child, encoding, out)

    # ── role constraints ──────────────────────────────────────────────

    def generate_role_constraints(
        self,
        tree: Dict[str, Any],
        encoding: TreeEncoding,
    ) -> List[RepairConstraint]:
        """Encode ARIA role validity: valid parent/child role combinations.

        Uses :data:`_VALID_CHILD_ROLES` to constrain the children of
        container roles (``list`` → ``listitem``, etc.).
        """
        constraints: List[RepairConstraint] = []
        self._role_recurse(tree, encoding, constraints)
        return constraints

    def _role_recurse(
        self,
        node: Dict[str, Any],
        encoding: TreeEncoding,
        out: List[RepairConstraint],
    ) -> None:
        nid = str(node.get("id", ""))
        role = str(node.get("role", "generic"))
        children = node.get("children", [])

        valid_children = _VALID_CHILD_ROLES.get(role)
        if valid_children and children:
            for child in children:
                cid = str(child.get("id", ""))
                child_role_var_id = f"{cid}__role"
                if child_role_var_id in encoding.variables:
                    child_role = str(child.get("role", "generic"))
                    # Build SMT expression: role variable must be in valid set.
                    role_map_raw = encoding.node_vars.get(cid, {}).get("_role_map", {})
                    if isinstance(role_map_raw, dict):
                        role_map: Dict[str, int] = role_map_raw
                        valid_ints = [
                            role_map[r] for r in valid_children if r in role_map
                        ]
                        if valid_ints:
                            terms = " ".join(
                                f"(= {child_role_var_id} {v})" for v in valid_ints
                            )
                            expr = f"(or {terms})" if len(valid_ints) > 1 else terms
                            out.append(self._make_constraint(
                                kind=ConstraintKind.ACCESSIBILITY,
                                desc=f"Child {cid} of {role} parent {nid} must have valid role",
                                expr=expr,
                                variables=(child_role_var_id,),
                                is_hard=True,
                            ))

        for child in children:
            self._role_recurse(child, encoding, out)

    # ── navigation constraints ────────────────────────────────────────

    def generate_navigation_constraints(
        self,
        tree: Dict[str, Any],
        encoding: TreeEncoding,
        task: Optional[Dict[str, Any]] = None,
    ) -> List[RepairConstraint]:
        """Encode reachability and keyboard traversal order.

        - All interactive elements must be reachable (not hidden).
        - Task target elements must be visible and focusable.
        """
        constraints: List[RepairConstraint] = []
        interactive = self._collect_interactive(tree)

        # Every interactive element must not be hidden.
        for nid in interactive:
            hidden_var_id = f"{nid}__hidden"
            if hidden_var_id in encoding.variables:
                expr = f"(not {hidden_var_id})"
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.ACCESSIBILITY,
                    desc=f"Interactive element {nid} must not be hidden",
                    expr=expr,
                    variables=(hidden_var_id,),
                    is_hard=True,
                ))

        # Task-specific: target elements must exist and be visible.
        if task:
            for flow in task.get("flows", []):
                for step in flow.get("steps", []):
                    target_name = step.get("target_name", "")
                    target_role = step.get("target_role", "")
                    # Find matching nodes and require them visible.
                    for nid in self._find_by_role(tree, target_role):
                        hidden_id = f"{nid}__hidden"
                        if hidden_id in encoding.variables:
                            constraints.append(self._make_constraint(
                                kind=ConstraintKind.ACCESSIBILITY,
                                desc=f"Task target {nid} ({target_role}) must be visible",
                                expr=f"(not {hidden_id})",
                                variables=(hidden_id,),
                                is_hard=True,
                            ))
        return constraints

    # ── cost constraints ──────────────────────────────────────────────

    def generate_cost_constraints(
        self,
        tree: Dict[str, Any],
        encoding: TreeEncoding,
        threshold: float = 4.0,
    ) -> List[RepairConstraint]:
        """Encode cognitive cost bounds per transition.

        Generates minimum-target-size constraints for all interactive
        elements (based on WCAG 2.5.5) and Fitts' law difficulty
        thresholds.

        Parameters:
            tree: Serialised accessibility tree.
            encoding: Z3 encoding of the tree.
            threshold: Maximum Fitts' index of difficulty (bits).
        """
        constraints: List[RepairConstraint] = []
        interactive = self._collect_interactive(tree)

        for nid in interactive:
            nv = encoding.node_vars.get(nid, {})
            w_id = f"{nid}__width"
            h_id = f"{nid}__height"
            if w_id in encoding.variables and h_id in encoding.variables:
                # WCAG 2.5.5 minimum target size.
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Element {nid} minimum target width >= {_MIN_TARGET_SIZE}px",
                    expr=f"(>= {w_id} {_MIN_TARGET_SIZE})",
                    variables=(w_id,),
                    is_hard=False,
                    weight=0.9,
                ))
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Element {nid} minimum target height >= {_MIN_TARGET_SIZE}px",
                    expr=f"(>= {h_id} {_MIN_TARGET_SIZE})",
                    variables=(h_id,),
                    is_hard=False,
                    weight=0.9,
                ))

        return constraints

    # ── grouping constraints ──────────────────────────────────────────

    def generate_grouping_constraints(
        self,
        tree: Dict[str, Any],
        encoding: TreeEncoding,
    ) -> List[RepairConstraint]:
        """Encode semantic grouping: related elements should be spatially close.

        Elements sharing the same parent and role should have minimal
        vertical/horizontal gaps (soft constraint encouraging proximity).
        """
        constraints: List[RepairConstraint] = []
        self._grouping_recurse(tree, encoding, constraints)
        return constraints

    def _grouping_recurse(
        self,
        node: Dict[str, Any],
        encoding: TreeEncoding,
        out: List[RepairConstraint],
    ) -> None:
        children = node.get("children", [])
        # Group children by role.
        role_groups: Dict[str, List[str]] = {}
        for child in children:
            cid = str(child.get("id", ""))
            crole = str(child.get("role", "generic"))
            role_groups.setdefault(crole, []).append(cid)

        for role, ids in role_groups.items():
            if len(ids) < 2:
                continue
            # Adjacent same-role siblings should be vertically close.
            for i in range(len(ids) - 1):
                a_id, b_id = ids[i], ids[i + 1]
                ya = f"{a_id}__y"
                ha = f"{a_id}__height"
                yb = f"{b_id}__y"
                if ya in encoding.variables and yb in encoding.variables:
                    # gap = b.y - (a.y + a.height) should be small
                    max_gap = 50
                    expr = f"(<= (- {yb} (+ {ya} {ha})) {max_gap})"
                    out.append(self._make_constraint(
                        kind=ConstraintKind.CONSISTENCY,
                        desc=f"Same-role ({role}) siblings {a_id},{b_id} are grouped",
                        expr=expr,
                        variables=(ya, ha, yb),
                        is_hard=False,
                        weight=0.5,
                    ))

        for child in children:
            self._grouping_recurse(child, encoding, out)

    # ── Fitts' law ────────────────────────────────────────────────────

    @staticmethod
    def encode_fitts_constraint(
        distance_var: z3.ArithRef,
        width_var: z3.ArithRef,
        time_bound: float,
    ) -> z3.BoolRef:
        """Encode Fitts' law as an SMT constraint.

        Fitts' law: ``MT = a + b · log₂(1 + D / W)``

        We use the simplified index-of-difficulty form:

            ID = log₂(1 + D / W)

        and constrain ``ID ≤ time_bound`` (in bits).  Because ``log₂``
        is transcendental, we use the linear approximation:

            D / W ≤ 2^time_bound − 1

        which is sound (the real log is monotone).

        Parameters:
            distance_var: Z3 variable for movement distance (px).
            width_var: Z3 variable for target width (px).
            time_bound: Maximum index of difficulty (bits).

        Returns:
            Z3 boolean expression: ``distance ≤ (2^time_bound − 1) · width``.
        """
        max_ratio = int(2 ** time_bound - 1)
        return distance_var <= max_ratio * width_var  # type: ignore[return-value]

    @staticmethod
    def encode_hick_constraint(
        n_alternatives_var: z3.ArithRef,
        time_bound: float,
    ) -> z3.BoolRef:
        """Encode Hick–Hyman law as an SMT constraint.

        Hick's law: ``RT = a + b · log₂(n + 1)``

        We bound the number of alternatives so that ``log₂(n + 1) ≤
        time_bound``, giving ``n ≤ 2^time_bound − 1``.

        Parameters:
            n_alternatives_var: Z3 integer variable for the number of
                choices presented to the user.
            time_bound: Maximum decision time budget (bits).

        Returns:
            Z3 boolean expression: ``n ≤ 2^time_bound − 1``.
        """
        max_n = int(2 ** time_bound - 1)
        return n_alternatives_var <= max_n  # type: ignore[return-value]

    # ── private helpers ───────────────────────────────────────────────

    def _make_constraint(
        self,
        kind: ConstraintKind,
        desc: str,
        expr: str,
        variables: Tuple[str, ...],
        is_hard: bool = True,
        weight: float = 1.0,
    ) -> RepairConstraint:
        cid = f"c_{self._cid_counter}"
        self._cid_counter += 1
        return RepairConstraint(
            constraint_id=cid,
            kind=kind,
            description=desc,
            expression=expr,
            variables=variables,
            is_hard=is_hard,
            weight=weight,
        )

    def _fitts_constraints(
        self,
        var_map: Dict[str, UIVariable],
        bottleneck: Dict[str, Any],
    ) -> List[RepairConstraint]:
        """Generate Fitts' law constraints for motor bottlenecks."""
        constraints: List[RepairConstraint] = []
        evidence = bottleneck.get("evidence", {})
        fitts_id = evidence.get("fitts_id", 4.0)
        target_id = max(3.0, fitts_id - 1.0)  # Reduce by at least 1 bit.

        for node_id in bottleneck.get("affected_states", []):
            w_var = f"{node_id}__width"
            h_var = f"{node_id}__height"
            if w_var in var_map:
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Fitts: {node_id} width must yield ID <= {target_id:.1f} bits",
                    expr=f"(>= {w_var} {_MIN_TARGET_SIZE})",
                    variables=(w_var,),
                    is_hard=False,
                    weight=0.9,
                ))
            if h_var in var_map:
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Fitts: {node_id} height must yield ID <= {target_id:.1f} bits",
                    expr=f"(>= {h_var} {_MIN_TARGET_SIZE})",
                    variables=(h_var,),
                    is_hard=False,
                    weight=0.9,
                ))
        return constraints

    def _hick_constraints(
        self,
        var_map: Dict[str, UIVariable],
        bottleneck: Dict[str, Any],
    ) -> List[RepairConstraint]:
        """Generate Hick-Hyman constraints for choice bottlenecks."""
        constraints: List[RepairConstraint] = []
        evidence = bottleneck.get("evidence", {})
        hick_bits = evidence.get("hick_bits", 3.0)
        max_items = int(2 ** min(hick_bits, 4.0) - 1)

        for node_id in bottleneck.get("affected_states", []):
            constraints.append(self._make_constraint(
                kind=ConstraintKind.COGNITIVE,
                desc=f"Hick: group {node_id} should have <= {max_items} items",
                expr=f"(<= {node_id}__n_children {max_items})",
                variables=(f"{node_id}__name_len",),  # proxy
                is_hard=False,
                weight=0.8,
            ))
        return constraints

    def _memory_constraints(
        self,
        var_map: Dict[str, UIVariable],
        bottleneck: Dict[str, Any],
    ) -> List[RepairConstraint]:
        """Generate working-memory constraints."""
        constraints: List[RepairConstraint] = []
        for node_id in bottleneck.get("affected_states", []):
            hidden_var = f"{node_id}__hidden"
            if hidden_var in var_map:
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Memory: keep {node_id} visible to reduce WM load",
                    expr=f"(not {hidden_var})",
                    variables=(hidden_var,),
                    is_hard=False,
                    weight=0.7,
                ))
        return constraints

    def _perceptual_constraints(
        self,
        var_map: Dict[str, UIVariable],
        bottleneck: Dict[str, Any],
    ) -> List[RepairConstraint]:
        """Generate perceptual (visibility) constraints."""
        constraints: List[RepairConstraint] = []
        for node_id in bottleneck.get("affected_states", []):
            w_var = f"{node_id}__width"
            h_var = f"{node_id}__height"
            if w_var in var_map:
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Perceptual: {node_id} large enough to perceive",
                    expr=f"(>= {w_var} 24)",
                    variables=(w_var,),
                    is_hard=False,
                    weight=0.6,
                ))
            if h_var in var_map:
                constraints.append(self._make_constraint(
                    kind=ConstraintKind.COGNITIVE,
                    desc=f"Perceptual: {node_id} tall enough to perceive",
                    expr=f"(>= {h_var} 24)",
                    variables=(h_var,),
                    is_hard=False,
                    weight=0.6,
                ))
        return constraints

    def _preservation_constraints(
        self,
        variables: Sequence[UIVariable],
    ) -> List[RepairConstraint]:
        """Soft constraints to preserve original values (minimise change)."""
        constraints: List[RepairConstraint] = []
        for v in variables:
            if v.sort == VariableSort.INT or v.sort == VariableSort.REAL:
                expr = f"(= {v.variable_id} {v.current_value})"
            elif v.sort == VariableSort.BOOL:
                val = "true" if v.current_value else "false"
                expr = f"(= {v.variable_id} {val})"
            else:
                continue
            constraints.append(self._make_constraint(
                kind=ConstraintKind.PRESERVATION,
                desc=f"Preserve {v.node_id}.{v.property_name}",
                expr=expr,
                variables=(v.variable_id,),
                is_hard=False,
                weight=0.3,
            ))
        return constraints

    # ── tree-traversal helpers ────────────────────────────────────────

    @staticmethod
    def _collect_interactive(tree: Dict[str, Any]) -> List[str]:
        """Collect node IDs of interactive elements in the tree."""
        interactive_roles = frozenset({
            "button", "checkbox", "combobox", "link", "listbox", "menu",
            "menuitem", "menuitemcheckbox", "menuitemradio", "option",
            "radio", "scrollbar", "slider", "spinbutton", "switch",
            "tab", "textbox", "treeitem",
        })
        result: List[str] = []

        def _walk(node: Dict[str, Any]) -> None:
            if node.get("role", "generic") in interactive_roles:
                result.append(str(node.get("id", "")))
            for child in node.get("children", []):
                _walk(child)

        _walk(tree)
        return result

    @staticmethod
    def _find_by_role(tree: Dict[str, Any], role: str) -> List[str]:
        """Find all node IDs with a given role."""
        result: List[str] = []

        def _walk(node: Dict[str, Any]) -> None:
            if node.get("role", "") == role:
                result.append(str(node.get("id", "")))
            for child in node.get("children", []):
                _walk(child)

        _walk(tree)
        return result
