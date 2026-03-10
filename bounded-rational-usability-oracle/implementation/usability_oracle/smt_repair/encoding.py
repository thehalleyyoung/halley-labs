"""
usability_oracle.smt_repair.encoding — UI structure encoding to Z3.

Maps accessibility tree nodes to Z3 variables and constraints, providing
the low-level bridge between the symbolic UI representation and the SMT
solver.  Each node's mutable properties (position, size, role, label,
visibility) become Z3 variables; spatial and structural relationships
become Z3 assertions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import z3

from usability_oracle.smt_repair.types import (
    UIVariable,
    VariableSort,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INT_LB = 0
_DEFAULT_INT_UB = 10_000
_DEFAULT_REAL_LB = 0.0
_DEFAULT_REAL_UB = 10_000.0

# Spatial relation tags used by encode_spatial_relation.
ABOVE = "above"
BELOW = "below"
LEFT_OF = "left_of"
RIGHT_OF = "right_of"
CONTAINS = "contains"
NON_OVERLAP = "non_overlap"


# ═══════════════════════════════════════════════════════════════════════════
# TreeEncoding — result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TreeEncoding:
    """Container holding the Z3 encoding of an accessibility tree.

    Attributes:
        variables: Mapping from ``variable_id`` to its Z3 expression.
        ui_variables: Mapping from ``variable_id`` to the source
            :class:`UIVariable` metadata.
        node_vars: Mapping from ``node_id`` to the dict of property-name
            → Z3 expression for that node.
        assertions: Accumulated Z3 assertions (domain bounds, etc.).
        enum_sorts: Named Z3 enum sorts created for string properties.
    """

    variables: Dict[str, z3.ExprRef] = field(default_factory=dict)
    ui_variables: Dict[str, UIVariable] = field(default_factory=dict)
    node_vars: Dict[str, Dict[str, z3.ExprRef]] = field(default_factory=dict)
    assertions: List[z3.BoolRef] = field(default_factory=list)
    enum_sorts: Dict[str, Tuple[z3.DatatypeSortRef, Dict[str, z3.ExprRef]]] = field(
        default_factory=dict
    )


# ═══════════════════════════════════════════════════════════════════════════
# Z3Encoder
# ═══════════════════════════════════════════════════════════════════════════

class Z3Encoder:
    """Encodes UI accessibility trees as Z3 variables and assertions.

    Usage::

        encoder = Z3Encoder()
        encoding = encoder.encode_tree(tree_dict)
        # encoding.variables  — Z3 variables
        # encoding.assertions — domain-bound constraints
    """

    def __init__(self) -> None:
        self._counter: int = 0

    # ── public API ────────────────────────────────────────────────────

    def encode_tree(self, tree: Dict[str, Any]) -> TreeEncoding:
        """Map an accessibility tree dict to Z3 variables.

        Recursively encodes every node in *tree* (which follows the
        canonical ``AccessibilityNode.to_dict()`` schema) into Z3
        variables for its mutable properties and adds domain-bound
        assertions.

        Parameters:
            tree: Serialised accessibility tree (root node dict with
                recursive ``"children"`` lists).

        Returns:
            A :class:`TreeEncoding` with all variables and assertions.
        """
        encoding = TreeEncoding()
        self._encode_subtree(tree, encoding)
        return encoding

    def encode_node(self, node: Dict[str, Any]) -> TreeEncoding:
        """Encode a single accessibility node to Z3 variables.

        Creates Z3 variables for position (``x``, ``y``), size
        (``width``, ``height``), ``role``, ``name``, and ``hidden``
        flag.

        Parameters:
            node: A single node dict (not recursed into children).

        Returns:
            A :class:`TreeEncoding` containing only this node's
            variables and assertions.
        """
        encoding = TreeEncoding()
        self._encode_single_node(node, encoding)
        return encoding

    def encode_integer_variable(
        self,
        name: str,
        bounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[z3.ArithRef, List[z3.BoolRef]]:
        """Create a bounded Z3 integer variable.

        Parameters:
            name: Variable name.
            bounds: ``(lower, upper)`` inclusive bounds; defaults to
                ``(0, 10000)``.

        Returns:
            ``(z3_var, [bound_assertions])``.
        """
        var = z3.Int(name)
        lb, ub = bounds if bounds is not None else (_DEFAULT_INT_LB, _DEFAULT_INT_UB)
        asserts: List[z3.BoolRef] = [var >= lb, var <= ub]
        return var, asserts

    def encode_real_variable(
        self,
        name: str,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[z3.ArithRef, List[z3.BoolRef]]:
        """Create a bounded Z3 real variable.

        Parameters:
            name: Variable name.
            bounds: ``(lower, upper)`` inclusive bounds; defaults to
                ``(0.0, 10000.0)``.

        Returns:
            ``(z3_var, [bound_assertions])``.
        """
        var = z3.Real(name)
        lb, ub = bounds if bounds is not None else (_DEFAULT_REAL_LB, _DEFAULT_REAL_UB)
        asserts: List[z3.BoolRef] = [var >= z3.RealVal(lb), var <= z3.RealVal(ub)]
        return var, asserts

    def encode_enum_variable(
        self,
        name: str,
        values: Sequence[str],
    ) -> Tuple[z3.ArithRef, List[z3.BoolRef], Dict[str, int]]:
        """Encode a finite-domain enumeration as a bounded integer.

        Each allowed string value is mapped to a consecutive integer
        starting from 0.  A domain constraint restricts the variable
        to ``[0, len(values)-1]``.

        Parameters:
            name: Variable name.
            values: Allowed string values.

        Returns:
            ``(z3_int_var, [domain_assertions], value_map)`` where
            *value_map* maps each string to its integer encoding.
        """
        value_map: Dict[str, int] = {v: i for i, v in enumerate(values)}
        var = z3.Int(name)
        asserts: List[z3.BoolRef] = [var >= 0, var <= len(values) - 1]
        return var, asserts, value_map

    def encode_spatial_relation(
        self,
        node_a_vars: Dict[str, z3.ExprRef],
        node_b_vars: Dict[str, z3.ExprRef],
        relation: str,
    ) -> z3.BoolRef:
        """Encode a spatial relationship between two encoded nodes.

        Supported *relation* values:

        - ``"above"``: *a* is entirely above *b*.
        - ``"below"``: *a* is entirely below *b*.
        - ``"left_of"``: *a* is entirely to the left of *b*.
        - ``"right_of"``: *a* is entirely to the right of *b*.
        - ``"contains"``: *a* contains *b* (bounding-box inclusion).
        - ``"non_overlap"``: *a* and *b* do not overlap.

        Parameters:
            node_a_vars: Z3 variable dict for node A (keys ``x``, ``y``,
                ``width``, ``height``).
            node_b_vars: Z3 variable dict for node B.
            relation: One of the relation tags above.

        Returns:
            A Z3 boolean expression encoding the relation.

        Raises:
            ValueError: On unknown *relation*.
        """
        ax, ay = node_a_vars["x"], node_a_vars["y"]
        aw, ah = node_a_vars["width"], node_a_vars["height"]
        bx, by = node_b_vars["x"], node_b_vars["y"]
        bw, bh = node_b_vars["width"], node_b_vars["height"]

        if relation == ABOVE:
            return ay + ah <= by  # type: ignore[return-value]
        if relation == BELOW:
            return by + bh <= ay  # type: ignore[return-value]
        if relation == LEFT_OF:
            return ax + aw <= bx  # type: ignore[return-value]
        if relation == RIGHT_OF:
            return bx + bw <= ax  # type: ignore[return-value]
        if relation == CONTAINS:
            return z3.And(
                ax <= bx,
                ay <= by,
                ax + aw >= bx + bw,
                ay + ah >= by + bh,
            )
        if relation == NON_OVERLAP:
            return z3.Or(
                ax + aw <= bx,
                bx + bw <= ax,
                ay + ah <= by,
                by + bh <= ay,
            )
        raise ValueError(f"Unknown spatial relation: {relation!r}")

    def encode_semantic_constraint(
        self,
        role_var: z3.ArithRef,
        role_map: Dict[str, int],
        required_children_roles: Sequence[str],
        children_role_vars: Sequence[z3.ArithRef],
    ) -> z3.BoolRef:
        """Encode that a parent with a given role must contain specific child roles.

        For example, a ``"list"`` must contain at least one ``"listitem"``.
        The encoding uses the integer-mapped role variables produced by
        :meth:`encode_enum_variable`.

        Parameters:
            role_var: Z3 integer variable for the parent's role.
            role_map: Mapping from role string to integer.
            required_children_roles: Roles that must appear among children.
            children_role_vars: Z3 integer variables for each child's role.

        Returns:
            Z3 boolean expression encoding the requirement.
        """
        if not required_children_roles or not children_role_vars:
            return z3.BoolVal(True)

        clauses: List[z3.BoolRef] = []
        for req_role in required_children_roles:
            if req_role not in role_map:
                continue
            req_val = role_map[req_role]
            # At least one child must have this role.
            child_has_role = z3.Or(*[cv == req_val for cv in children_role_vars])
            clauses.append(child_has_role)

        if not clauses:
            return z3.BoolVal(True)
        return z3.And(*clauses)

    def decode_model(
        self,
        z3_model: z3.ModelRef,
        encoding: TreeEncoding,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract a solution from a Z3 model back to property values.

        Iterates over all encoded UI variables and reads their assigned
        values from *z3_model*.

        Parameters:
            z3_model: A satisfying Z3 model.
            encoding: The :class:`TreeEncoding` that was used to build
                the solver assertions.

        Returns:
            Mapping from ``node_id`` to ``{property_name: value}`` for
            every encoded node.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for var_id, z3_var in encoding.variables.items():
            ui_var = encoding.ui_variables.get(var_id)
            if ui_var is None:
                continue

            node_id = ui_var.node_id
            prop = ui_var.property_name

            val = z3_model.evaluate(z3_var, model_completion=True)
            py_val = self._z3_val_to_python(val, ui_var)

            result.setdefault(node_id, {})[prop] = py_val

        return result

    # ── private helpers ───────────────────────────────────────────────

    def _encode_subtree(
        self,
        node: Dict[str, Any],
        encoding: TreeEncoding,
    ) -> None:
        """Recursively encode *node* and its children."""
        self._encode_single_node(node, encoding)
        for child in node.get("children", []):
            self._encode_subtree(child, encoding)

    def _encode_single_node(
        self,
        node: Dict[str, Any],
        encoding: TreeEncoding,
    ) -> None:
        """Encode one node's properties as Z3 variables."""
        nid = str(node.get("id", f"node_{self._counter}"))
        self._counter += 1
        node_vars: Dict[str, z3.ExprRef] = {}

        # Bounding box (x, y, width, height) as integers.
        bbox = node.get("bounding_box", {})
        for prop, default in [("x", 0), ("y", 0), ("width", 100), ("height", 40)]:
            var_id = f"{nid}__{prop}"
            current = int(bbox.get(prop, default))
            var, asserts = self.encode_integer_variable(var_id, (0, _DEFAULT_INT_UB))
            encoding.variables[var_id] = var
            encoding.ui_variables[var_id] = UIVariable(
                variable_id=var_id,
                node_id=nid,
                property_name=prop,
                sort=VariableSort.INT,
                current_value=current,
                lower_bound=0,
                upper_bound=_DEFAULT_INT_UB,
            )
            encoding.assertions.extend(asserts)
            node_vars[prop] = var

        # Visibility (hidden flag) as a boolean.
        state = node.get("state", {})
        hidden_id = f"{nid}__hidden"
        hidden_var = z3.Bool(hidden_id)
        hidden_val = bool(state.get("hidden", False))
        encoding.variables[hidden_id] = hidden_var
        encoding.ui_variables[hidden_id] = UIVariable(
            variable_id=hidden_id,
            node_id=nid,
            property_name="hidden",
            sort=VariableSort.BOOL,
            current_value=hidden_val,
        )
        node_vars["hidden"] = hidden_var

        # Role as an integer-encoded enum.
        role_str = str(node.get("role", "generic"))
        role_id = f"{nid}__role"
        role_values = _COMMON_ROLES
        role_var, role_asserts, role_map = self.encode_enum_variable(role_id, role_values)
        encoding.variables[role_id] = role_var
        cur_role_int = role_map.get(role_str, 0)
        encoding.ui_variables[role_id] = UIVariable(
            variable_id=role_id,
            node_id=nid,
            property_name="role",
            sort=VariableSort.STRING,
            current_value=role_str,
            lower_bound=0,
            upper_bound=len(role_values) - 1,
            allowed_values=frozenset(role_values),
        )
        encoding.assertions.extend(role_asserts)
        node_vars["role"] = role_var
        node_vars["_role_map"] = role_map  # type: ignore[assignment]

        # Name / label length as an integer proxy.
        name_str = str(node.get("name", ""))
        name_len_id = f"{nid}__name_len"
        name_var, name_asserts = self.encode_integer_variable(name_len_id, (0, 500))
        encoding.variables[name_len_id] = name_var
        encoding.ui_variables[name_len_id] = UIVariable(
            variable_id=name_len_id,
            node_id=nid,
            property_name="name_len",
            sort=VariableSort.INT,
            current_value=len(name_str),
            lower_bound=0,
            upper_bound=500,
        )
        encoding.assertions.extend(name_asserts)
        node_vars["name_len"] = name_var

        encoding.node_vars[nid] = node_vars

    @staticmethod
    def _z3_val_to_python(val: z3.ExprRef, ui_var: UIVariable) -> Any:
        """Convert a Z3 value to a Python scalar."""
        if ui_var.sort == VariableSort.BOOL:
            return bool(z3.is_true(val))
        if ui_var.sort == VariableSort.INT:
            try:
                return int(str(val))
            except (ValueError, TypeError):
                return ui_var.current_value
        if ui_var.sort == VariableSort.REAL:
            try:
                s = str(val)
                if "/" in s:
                    num, den = s.split("/")
                    return float(num) / float(den)
                return float(s)
            except (ValueError, TypeError):
                return ui_var.current_value
        if ui_var.sort == VariableSort.STRING:
            # Role was integer-encoded; return the int for now.
            try:
                return int(str(val))
            except (ValueError, TypeError):
                return ui_var.current_value
        return str(val)


# ---------------------------------------------------------------------------
# Common ARIA roles
# ---------------------------------------------------------------------------

_COMMON_ROLES: List[str] = [
    "generic",
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
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "tablist",
    "textbox",
    "tree",
    "treegrid",
    "treeitem",
    "banner",
    "complementary",
    "contentinfo",
    "form",
    "main",
    "navigation",
    "region",
    "search",
    "article",
    "heading",
    "list",
    "listitem",
    "table",
    "cell",
    "img",
    "section",
    "note",
    "status",
    "alert",
    "progressbar",
    "alertdialog",
    "dialog",
    "tooltip",
    "separator",
    "toolbar",
    "group",
    "document",
    "row",
    "rowgroup",
    "columnheader",
    "rowheader",
]
