"""
usability_oracle.taskspec.inference — Auto-infer task specifications from
accessibility trees.

Given an :class:`AccessibilityTree` (or a duck-typed equivalent), the
:class:`TaskInferrer` detects common UI patterns and emits candidate
:class:`TaskSpec` objects covering:

* Form filling + submission
* Navigation (menus, breadcrumbs, tabs)
* Search (search box → results → selection)
* Selection tasks (lists, tables, combo boxes)
* Dialog interaction (modal open → interact → confirm/cancel)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from usability_oracle.taskspec.models import TaskFlow, TaskSpec, TaskStep


# ---------------------------------------------------------------------------
# Minimal protocol for accessibility tree nodes
# ---------------------------------------------------------------------------

class AccessibilityNode(Protocol):
    """Protocol for a single accessibility tree node."""

    @property
    def node_id(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def children(self) -> Sequence["AccessibilityNode"]: ...

    @property
    def properties(self) -> Dict[str, Any]: ...


class AccessibilityTree(Protocol):
    """Protocol for an accessibility tree."""

    @property
    def root(self) -> AccessibilityNode: ...

    def find_by_role(self, role: str) -> List[AccessibilityNode]: ...

    def find_by_name(self, name: str) -> List[AccessibilityNode]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _node_matches_role(node: Any, roles: Set[str]) -> bool:
    """Check if *node* has a role matching one of *roles*."""
    return getattr(node, "role", "").lower() in roles


def _get_node_name(node: Any) -> str:
    name = getattr(node, "name", "")
    if not name:
        props = getattr(node, "properties", {})
        name = props.get("aria-label", "") or props.get("title", "")
    return name


def _collect_descendants(node: Any, *, max_depth: int = 10) -> List[Any]:
    """Collect all descendant nodes up to *max_depth* levels."""
    if max_depth <= 0:
        return []
    result: List[Any] = []
    for child in getattr(node, "children", []):
        result.append(child)
        result.extend(_collect_descendants(child, max_depth=max_depth - 1))
    return result


def _find_by_role_recursive(node: Any, roles: Set[str], *, max_depth: int = 15) -> List[Any]:
    """Find all descendants matching one of *roles*."""
    matches: List[Any] = []
    for desc in _collect_descendants(node, max_depth=max_depth):
        if _node_matches_role(desc, roles):
            matches.append(desc)
    return matches


# ---------------------------------------------------------------------------
# TaskInferrer
# ---------------------------------------------------------------------------


class TaskInferrer:
    """Infer :class:`TaskSpec` objects from an accessibility tree.

    Usage::

        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(tree)
    """

    # Roles considered as form inputs
    FORM_INPUT_ROLES: Set[str] = {
        "textfield", "textbox", "input", "combobox", "listbox",
        "checkbox", "radio", "spinbutton", "slider", "searchbox",
        "textarea", "switch",
    }

    # Roles considered as buttons / submit triggers
    BUTTON_ROLES: Set[str] = {"button", "link", "menuitem", "submit"}

    # Roles for navigation structures
    NAV_ROLES: Set[str] = {"navigation", "menubar", "menu", "tablist", "tab", "breadcrumb"}

    # Roles for search patterns
    SEARCH_ROLES: Set[str] = {"searchbox", "search", "combobox"}

    # Roles for selection containers
    SELECTION_CONTAINER_ROLES: Set[str] = {"list", "listbox", "table", "grid", "tree", "treegrid"}

    # Roles for dialog patterns
    DIALOG_ROLES: Set[str] = {"dialog", "alertdialog", "modal"}

    def __init__(self, *, min_form_fields: int = 2, max_form_fields: int = 30) -> None:
        self._min_form_fields = min_form_fields
        self._max_form_fields = max_form_fields

    # -- public API ----------------------------------------------------------

    def infer_from_tree(self, tree: Any) -> List[TaskSpec]:
        """Auto-infer common task patterns from an accessibility tree.

        Parameters
        ----------
        tree : AccessibilityTree
            An accessibility tree (or any object with ``root``,
            ``find_by_role``, and ``find_by_name`` methods).

        Returns
        -------
        list[TaskSpec]
            Candidate task specifications, one per detected pattern.
        """
        specs: List[TaskSpec] = []
        specs.extend(self._detect_form_tasks(tree))
        specs.extend(self._detect_navigation_tasks(tree))
        specs.extend(self._detect_search_tasks(tree))
        specs.extend(self._detect_selection_tasks(tree))
        specs.extend(self._detect_dialog_tasks(tree))
        return specs

    # -- form detection ------------------------------------------------------

    def _detect_form_tasks(self, tree: Any) -> List[TaskSpec]:
        """Detect form-filling tasks.

        Heuristic: find a group of nearby input fields followed by or
        containing a submit button.  Generate a TaskSpec that fills each
        field and clicks submit.
        """
        root = getattr(tree, "root", tree)
        # Find all form-like containers
        form_containers = self._find_form_containers(root)
        specs: List[TaskSpec] = []

        for container, inputs, buttons in form_containers:
            if len(inputs) < self._min_form_fields:
                continue
            if len(inputs) > self._max_form_fields:
                continue

            container_name = _get_node_name(container) or "form"

            steps: List[TaskStep] = []
            for inp in inputs:
                inp_name = _get_node_name(inp) or inp.role
                inp_role = getattr(inp, "role", "textfield").lower()

                # Click to focus
                steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role=inp_role,
                    target_name=inp_name,
                    description=f"Focus {inp_name} field",
                ))

                # Type or select depending on role
                if inp_role in ("combobox", "listbox", "checkbox", "radio", "switch"):
                    steps.append(TaskStep(
                        step_id=_gen_id("step"),
                        action_type="select",
                        target_role=inp_role,
                        target_name=inp_name,
                        input_value="<selection>",
                        description=f"Select value in {inp_name}",
                    ))
                else:
                    steps.append(TaskStep(
                        step_id=_gen_id("step"),
                        action_type="type",
                        target_role=inp_role,
                        target_name=inp_name,
                        input_value=f"<{inp_name}_value>",
                        description=f"Enter value for {inp_name}",
                    ))

            # Add submit button click
            if buttons:
                btn = buttons[0]
                btn_name = _get_node_name(btn) or "Submit"
                steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role=getattr(btn, "role", "button").lower(),
                    target_name=btn_name,
                    description=f"Click {btn_name}",
                ))

            flow = TaskFlow(
                flow_id=_gen_id("flow"),
                name=f"fill_{container_name}",
                steps=steps,
                success_criteria=[f"form_{container_name}_submitted"],
            )
            specs.append(TaskSpec(
                spec_id=_gen_id("spec"),
                name=f"form_{container_name}",
                description=f"Fill and submit the {container_name} form",
                flows=[flow],
                metadata={"inferred": True, "pattern": "form"},
            ))

        return specs

    def _find_form_containers(self, root: Any) -> List[Tuple[Any, List[Any], List[Any]]]:
        """Find form containers: (container_node, input_nodes, button_nodes)."""
        results: List[Tuple[Any, List[Any], List[Any]]] = []

        # Look for explicit form roles
        all_descendants = _collect_descendants(root)
        form_nodes = [n for n in all_descendants
                      if getattr(n, "role", "").lower() in ("form", "group")]

        if not form_nodes:
            # Fallback: treat root as a single container
            form_nodes = [root]

        for container in form_nodes:
            descs = _collect_descendants(container)
            inputs = [n for n in descs if _node_matches_role(n, self.FORM_INPUT_ROLES)]
            buttons = [n for n in descs if _node_matches_role(n, self.BUTTON_ROLES)]
            if inputs:
                results.append((container, inputs, buttons))

        return results

    # -- navigation detection ------------------------------------------------

    def _detect_navigation_tasks(self, tree: Any) -> List[TaskSpec]:
        """Detect navigation tasks from menus, tabs, breadcrumbs."""
        root = getattr(tree, "root", tree)
        nav_nodes = _find_by_role_recursive(root, self.NAV_ROLES)
        specs: List[TaskSpec] = []

        for nav in nav_nodes:
            nav_name = _get_node_name(nav) or getattr(nav, "role", "nav")
            # Collect clickable children (links, tabs, menu items)
            clickables = _find_by_role_recursive(nav, {"link", "tab", "menuitem", "menuitemradio"})
            if not clickables:
                continue

            for target in clickables:
                target_name = _get_node_name(target) or "item"
                steps = [TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role=getattr(target, "role", "link").lower(),
                    target_name=target_name,
                    description=f"Navigate to {target_name}",
                    postconditions=[f"page_contains_{target_name.lower().replace(' ', '_')}"],
                )]
                flow = TaskFlow(
                    flow_id=_gen_id("flow"),
                    name=f"navigate_to_{target_name.lower().replace(' ', '_')}",
                    steps=steps,
                    success_criteria=[f"navigated_to_{target_name}"],
                )
                specs.append(TaskSpec(
                    spec_id=_gen_id("spec"),
                    name=f"nav_{target_name}",
                    description=f"Navigate to {target_name} via {nav_name}",
                    flows=[flow],
                    metadata={"inferred": True, "pattern": "navigation"},
                ))

        return specs

    # -- search detection ----------------------------------------------------

    def _detect_search_tasks(self, tree: Any) -> List[TaskSpec]:
        """Detect search-and-select tasks.

        Pattern: search box → type query → wait for results → select result.
        """
        root = getattr(tree, "root", tree)
        search_boxes = _find_by_role_recursive(root, self.SEARCH_ROLES)
        specs: List[TaskSpec] = []

        for sb in search_boxes:
            sb_name = _get_node_name(sb) or "Search"
            steps = [
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role=getattr(sb, "role", "searchbox").lower(),
                    target_name=sb_name,
                    description=f"Click {sb_name}",
                ),
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="type",
                    target_role=getattr(sb, "role", "searchbox").lower(),
                    target_name=sb_name,
                    input_value="<search_query>",
                    description=f"Type search query in {sb_name}",
                ),
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="wait",
                    target_role="region",
                    target_name="Search Results",
                    description="Wait for search results to appear",
                    timeout=5.0,
                ),
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="link",
                    target_name="<first_result>",
                    description="Select first search result",
                ),
            ]
            flow = TaskFlow(
                flow_id=_gen_id("flow"),
                name="search_and_select",
                steps=steps,
                success_criteria=["result_page_loaded"],
            )
            specs.append(TaskSpec(
                spec_id=_gen_id("spec"),
                name=f"search_{sb_name.lower().replace(' ', '_')}",
                description=f"Search using {sb_name} and select a result",
                flows=[flow],
                metadata={"inferred": True, "pattern": "search"},
            ))

        return specs

    # -- selection detection -------------------------------------------------

    def _detect_selection_tasks(self, tree: Any) -> List[TaskSpec]:
        """Detect selection tasks from lists, tables, and grids."""
        root = getattr(tree, "root", tree)
        containers = _find_by_role_recursive(root, self.SELECTION_CONTAINER_ROLES)
        specs: List[TaskSpec] = []

        for container in containers:
            cont_name = _get_node_name(container) or getattr(container, "role", "list")
            cont_role = getattr(container, "role", "list").lower()

            # Find selectable items inside
            item_roles = {"listitem", "row", "option", "treeitem", "gridcell", "cell"}
            items = _find_by_role_recursive(container, item_roles)
            if not items:
                continue

            steps = [
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="scroll",
                    target_role=cont_role,
                    target_name=cont_name,
                    description=f"Scroll {cont_name} to find target item",
                    optional=True,
                ),
                TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role=getattr(items[0], "role", "listitem").lower(),
                    target_name="<target_item>",
                    description=f"Select item from {cont_name}",
                ),
            ]
            flow = TaskFlow(
                flow_id=_gen_id("flow"),
                name=f"select_from_{cont_name.lower().replace(' ', '_')}",
                steps=steps,
                success_criteria=[f"item_selected_from_{cont_name}"],
            )
            specs.append(TaskSpec(
                spec_id=_gen_id("spec"),
                name=f"select_{cont_name}",
                description=f"Select an item from {cont_name}",
                flows=[flow],
                metadata={
                    "inferred": True,
                    "pattern": "selection",
                    "item_count": len(items),
                },
            ))

        return specs

    # -- dialog detection ----------------------------------------------------

    def _detect_dialog_tasks(self, tree: Any) -> List[TaskSpec]:
        """Detect modal dialog tasks.

        Pattern: dialog appears → interact with contents → confirm or cancel.
        """
        root = getattr(tree, "root", tree)
        dialogs = _find_by_role_recursive(root, self.DIALOG_ROLES)
        specs: List[TaskSpec] = []

        for dialog in dialogs:
            dlg_name = _get_node_name(dialog) or "Dialog"

            # Find interactive elements inside the dialog
            inputs = _find_by_role_recursive(dialog, self.FORM_INPUT_ROLES)
            buttons = _find_by_role_recursive(dialog, self.BUTTON_ROLES)

            # Confirm / cancel buttons heuristic
            confirm_btn = None
            cancel_btn = None
            for btn in buttons:
                btn_name = _get_node_name(btn).lower()
                if any(w in btn_name for w in ("ok", "confirm", "submit", "save", "yes", "accept")):
                    confirm_btn = btn
                elif any(w in btn_name for w in ("cancel", "close", "no", "dismiss")):
                    cancel_btn = btn

            # --- Confirm path ---
            confirm_steps: List[TaskStep] = []
            for inp in inputs:
                inp_name = _get_node_name(inp) or "field"
                confirm_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role=getattr(inp, "role", "textfield").lower(),
                    target_name=inp_name,
                    description=f"Focus {inp_name}",
                ))
                confirm_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="type",
                    target_role=getattr(inp, "role", "textfield").lower(),
                    target_name=inp_name,
                    input_value=f"<{inp_name}_value>",
                    description=f"Fill {inp_name}",
                ))

            if confirm_btn:
                confirm_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="button",
                    target_name=_get_node_name(confirm_btn) or "Confirm",
                    description="Confirm dialog",
                ))
            confirm_flow = TaskFlow(
                flow_id=_gen_id("flow"),
                name=f"confirm_{dlg_name.lower().replace(' ', '_')}",
                steps=confirm_steps,
                success_criteria=["dialog_closed", "action_confirmed"],
            )

            # --- Cancel path ---
            cancel_steps: List[TaskStep] = []
            if cancel_btn:
                cancel_steps.append(TaskStep(
                    step_id=_gen_id("step"),
                    action_type="click",
                    target_role="button",
                    target_name=_get_node_name(cancel_btn) or "Cancel",
                    description="Cancel dialog",
                ))
            cancel_flow = TaskFlow(
                flow_id=_gen_id("flow"),
                name=f"cancel_{dlg_name.lower().replace(' ', '_')}",
                steps=cancel_steps,
                success_criteria=["dialog_closed"],
            )

            flows = [confirm_flow]
            if cancel_steps:
                flows.append(cancel_flow)

            specs.append(TaskSpec(
                spec_id=_gen_id("spec"),
                name=f"dialog_{dlg_name}",
                description=f"Interact with {dlg_name} dialog",
                flows=flows,
                metadata={
                    "inferred": True,
                    "pattern": "dialog",
                    "has_inputs": len(inputs) > 0,
                    "has_confirm": confirm_btn is not None,
                    "has_cancel": cancel_btn is not None,
                },
            ))

        return specs
