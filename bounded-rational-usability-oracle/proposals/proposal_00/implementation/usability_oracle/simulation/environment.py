"""
usability_oracle.simulation.environment — UI environment for simulation.

Provides a simulated UI environment that an agent can interact with,
tracking state transitions, available actions, and goal completion.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from usability_oracle.accessibility.models import AccessibilityNode, AccessibilityTree, BoundingBox


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

@dataclass
class UIState:
    """Current state of the UI environment."""
    id: str
    focused_element: Optional[str] = None
    visible_elements: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    form_values: dict[str, str] = field(default_factory=dict)
    navigation_path: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    elapsed_time: float = 0.0
    step_count: int = 0


@dataclass
class UIAction:
    """An available action in the UI environment."""
    id: str
    name: str
    action_type: str  # click, type, scroll, navigate, etc.
    target_element: str
    target_x: float = 0.0
    target_y: float = 0.0
    target_width: float = 0.0
    target_height: float = 0.0
    n_alternatives: int = 1
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "action_type": self.action_type,
            "target_element": self.target_element,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "n_alternatives": self.n_alternatives,
        }


# ---------------------------------------------------------------------------
# UIEnvironment
# ---------------------------------------------------------------------------

class UIEnvironment:
    """Simulated UI environment for agent interaction.

    Wraps an accessibility tree and provides action-based interaction.
    """

    def __init__(
        self,
        tree: AccessibilityTree,
        goal_elements: list[str] | None = None,
        task_steps: list[str] | None = None,
    ) -> None:
        self._tree = tree
        self._goal_elements = set(goal_elements or [])
        self._task_steps = task_steps or []
        self._state = UIState(id="initial")
        self._action_handlers: dict[str, Callable] = {}
        self._interactive_roles = {"button", "link", "textfield", "checkbox", "radio",
                                    "menuitem", "tab", "combobox", "slider"}
        self._init_visible_elements()

    def _init_visible_elements(self) -> None:
        """Populate initial visible elements."""
        visible = []
        def _walk(node: AccessibilityNode) -> None:
            state = node.state
            if state and state.hidden:
                return
            visible.append(node.id)
            for child in node.children:
                _walk(child)
        _walk(self._tree.root)
        self._state.visible_elements = visible

    # ------------------------------------------------------------------
    # Available actions
    # ------------------------------------------------------------------

    def get_available_actions(self) -> list[UIAction]:
        """Get all currently available actions."""
        actions: list[UIAction] = []

        for node_id in self._state.visible_elements:
            node = self._tree.node_index.get(node_id)
            if not node:
                continue

            role = node.role.lower() if isinstance(node.role, str) else str(node.role).lower()
            if role not in self._interactive_roles:
                continue

            state = node.state
            if state and state.disabled:
                continue

            bbox = node.bounding_box or BoundingBox(x=0, y=0, width=20, height=20)

            # Determine action type based on role
            if role in ("button", "link", "menuitem", "tab"):
                action_type = "click"
            elif role == "textfield":
                action_type = "type"
            elif role in ("checkbox", "radio"):
                action_type = "toggle"
            elif role == "combobox":
                action_type = "select"
            elif role == "slider":
                action_type = "adjust"
            else:
                action_type = "click"

            # Count sibling alternatives
            parent = self._tree.node_index.get(getattr(node, "parent_id", ""))
            n_alternatives = 1
            if parent:
                n_alternatives = sum(
                    1 for c in parent.children
                    if (c.role.lower() if isinstance(c.role, str) else str(c.role).lower()) in self._interactive_roles
                )

            actions.append(UIAction(
                id=node.id,
                name=node.name,
                action_type=action_type,
                target_element=node.id,
                target_x=bbox.x,
                target_y=bbox.y,
                target_width=bbox.width,
                target_height=bbox.height,
                n_alternatives=n_alternatives,
            ))

        return actions

    # ------------------------------------------------------------------
    # Execute action
    # ------------------------------------------------------------------

    def step(self, action: UIAction) -> UIState:
        """Execute an action and return the new state."""
        self._state.step_count += 1

        # Update focused element
        self._state.focused_element = action.target_element

        # Track completed steps
        if action.target_element in self._goal_elements:
            self._state.completed_steps.append(action.target_element)

        # Update navigation path
        self._state.navigation_path.append(action.target_element)

        # Handle form inputs
        if action.action_type == "type":
            self._state.form_values[action.target_element] = f"value_{self._state.step_count}"

        # Handle toggles
        if action.action_type == "toggle":
            node = self._tree.node_index.get(action.target_element)
            if node and node.state:
                node.state.checked = not (node.state.checked or False)

        # Update visible elements (simulate reveals)
        if action.action_type == "click":
            node = self._tree.node_index.get(action.target_element)
            if node and node.state and node.state.expanded is False:
                node.state.expanded = True
                for child in node.children:
                    if child.id not in self._state.visible_elements:
                        self._state.visible_elements.append(child.id)

        return copy.copy(self._state)

    # ------------------------------------------------------------------
    # Goal checking
    # ------------------------------------------------------------------

    def is_goal_reached(self) -> bool:
        """Check if the task goal has been reached."""
        if not self._goal_elements:
            return False
        return self._goal_elements.issubset(set(self._state.completed_steps))

    def task_progress(self) -> float:
        """Return task completion progress (0-1)."""
        if not self._task_steps:
            if not self._goal_elements:
                return 0.0
            return len(set(self._state.completed_steps) & self._goal_elements) / len(self._goal_elements)
        completed = sum(1 for s in self._task_steps if s in self._state.completed_steps)
        return completed / len(self._task_steps)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> UIState:
        return self._state

    @property
    def tree(self) -> AccessibilityTree:
        return self._tree

    def reset(self) -> None:
        """Reset environment to initial state."""
        self._state = UIState(id="initial")
        self._init_visible_elements()
