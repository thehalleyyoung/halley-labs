"""
usability_oracle.goms.decomposition — Hierarchical task decomposition.

Automatic task decomposition from accessibility trees: goal inference,
operator assignment, method construction for common UI patterns (form
filling, navigation, search+select, drag-and-drop, multi-step wizards),
hierarchical goal tree construction, and goal conflict detection.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from usability_oracle.core.types import BoundingBox, Point2D
from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    OperatorType,
)
from usability_oracle.goms.klm import (
    KLMConfig,
    make_button_press,
    make_button_release,
    make_homing,
    make_keystroke,
    make_mental,
    make_pointing,
    make_system_response,
)


# ═══════════════════════════════════════════════════════════════════════════
# UI pattern enumeration
# ═══════════════════════════════════════════════════════════════════════════

_FORM_ROLES: FrozenSet[str] = frozenset({
    "textfield", "checkbox", "radio", "combobox", "slider",
})
_NAVIGATION_ROLES: FrozenSet[str] = frozenset({
    "link", "menuitem", "tab", "treeitem",
})
_ACTION_ROLES: FrozenSet[str] = frozenset({
    "button",
})


@dataclass
class UIPattern:
    """A recognized UI interaction pattern."""

    pattern_type: str
    """One of: form_fill, navigation, search_select, drag_drop, wizard, click."""
    node_ids: Tuple[str, ...]
    """Accessibility node IDs involved in this pattern."""
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Goal conflict
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GoalConflict:
    """A detected conflict between two goals."""

    goal_a_id: str
    goal_b_id: str
    conflict_type: str
    """One of: resource_conflict, ordering_conflict, mutex_conflict."""
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Node info extraction
# ═══════════════════════════════════════════════════════════════════════════

def _node_role(node: Any) -> str:
    return getattr(node, "role", "").lower()


def _node_id(node: Any) -> str:
    return getattr(node, "node_id", str(uuid.uuid4())[:8])


def _node_name(node: Any) -> str:
    return getattr(node, "name", "")


def _node_bounds(node: Any) -> Optional[BoundingBox]:
    return getattr(node, "bounds", None)


def _node_children(node: Any) -> Sequence[Any]:
    return getattr(node, "children", [])


# ═══════════════════════════════════════════════════════════════════════════
# Pattern recognition from accessibility tree
# ═══════════════════════════════════════════════════════════════════════════

def recognize_patterns(tree: Any) -> List[UIPattern]:
    """Identify UI interaction patterns from an accessibility tree.

    Scans the tree for structural groupings of interactive elements
    and classifies them into known patterns.

    Returns
    -------
    list[UIPattern]
        Detected UI patterns in tree traversal order.
    """
    patterns: List[UIPattern] = []
    root = tree.root

    def _walk(node: Any, form_ctx: List[str], nav_ctx: List[str]) -> None:
        role = _node_role(node)
        nid = _node_id(node)

        if role == "form":
            # Collect all form fields within
            fields = _collect_interactive_descendants(node)
            if fields:
                patterns.append(UIPattern(
                    pattern_type="form_fill",
                    node_ids=tuple(fields),
                    description=f"Form: {_node_name(node) or 'unnamed'}",
                ))
            return  # Don't recurse further into form children

        if role in ("navigation", "menu", "toolbar", "tree"):
            items = _collect_interactive_descendants(node)
            if items:
                patterns.append(UIPattern(
                    pattern_type="navigation",
                    node_ids=tuple(items),
                    description=f"Navigation: {_node_name(node) or role}",
                ))
            return

        if role == "search":
            items = _collect_interactive_descendants(node)
            if items:
                patterns.append(UIPattern(
                    pattern_type="search_select",
                    node_ids=tuple(items),
                    description=f"Search: {_node_name(node) or 'search'}",
                ))
            return

        if role == "dialog":
            # Dialogs often represent wizard steps
            items = _collect_interactive_descendants(node)
            if items:
                patterns.append(UIPattern(
                    pattern_type="wizard",
                    node_ids=tuple(items),
                    description=f"Dialog/Wizard: {_node_name(node) or 'dialog'}",
                ))
            return

        # Individual interactive elements not inside a pattern container
        if role in _FORM_ROLES | _NAVIGATION_ROLES | _ACTION_ROLES:
            patterns.append(UIPattern(
                pattern_type="click" if role in _ACTION_ROLES else (
                    "form_fill" if role in _FORM_ROLES else "navigation"
                ),
                node_ids=(nid,),
                description=f"{role}: {_node_name(node) or nid}",
            ))

        for child in _node_children(node):
            _walk(child, form_ctx, nav_ctx)

    _walk(root, [], [])
    return patterns


def _collect_interactive_descendants(node: Any) -> List[str]:
    """Collect node IDs of all interactive descendants."""
    result: List[str] = []

    def _recurse(n: Any) -> None:
        role = _node_role(n)
        if role in _FORM_ROLES | _NAVIGATION_ROLES | _ACTION_ROLES:
            result.append(_node_id(n))
        for child in _node_children(n):
            _recurse(child)

    for child in _node_children(node):
        _recurse(child)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Operator assignment based on element type
# ═══════════════════════════════════════════════════════════════════════════

def operators_for_node(
    node: Any,
    cursor: Point2D,
    *,
    config: KLMConfig = KLMConfig(),
) -> Tuple[List[GomsOperator], Point2D]:
    """Generate KLM operators for interacting with a node.

    Parameters
    ----------
    node : AccessibilityNode
        The target node.
    cursor : Point2D
        Current cursor position.
    config : KLMConfig
        KLM calibration.

    Returns
    -------
    tuple[list[GomsOperator], Point2D]
        Operators and updated cursor position.
    """
    role = _node_role(node)
    nid = _node_id(node)
    bounds = _node_bounds(node)
    ops: List[GomsOperator] = []

    if bounds is not None:
        center = bounds.center
        width = bounds.width
    else:
        center = Point2D(cursor.x + 80.0, cursor.y)
        width = 60.0

    # Point to element
    ops.append(make_pointing(
        source=cursor,
        target_center=center,
        target_width_px=width,
        config=config,
        target_id=nid,
        target_bounds=bounds,
    ))
    new_cursor = center

    if role in ("button", "link", "menuitem", "tab", "treeitem", "listitem"):
        ops.append(make_button_press(config=config, target_id=nid))

    elif role == "textfield":
        ops.append(make_button_press(
            config=config, target_id=nid,
            description="Click text field",
        ))
        ops.append(make_homing(config=config, description="Home hand to keyboard"))
        # Type average field content (8 characters)
        for _ in range(8):
            ops.append(make_keystroke(config=config, target_id=nid))
        ops.append(make_homing(config=config, description="Home hand to mouse"))

    elif role in ("checkbox", "radio"):
        ops.append(make_button_press(config=config, target_id=nid))

    elif role == "combobox":
        # Click to open, wait, then select
        ops.append(make_button_press(
            config=config, target_id=nid,
            description="Open dropdown",
        ))
        ops.append(make_system_response(config=config, description="Dropdown opens"))
        ops.append(make_mental(config=config, n_choices=5, description="Scan choices"))
        ops.append(make_pointing(
            source=center,
            target_center=Point2D(center.x, center.y + 40),
            target_width_px=width,
            config=config,
            target_id=nid,
        ))
        ops.append(make_button_press(
            config=config, target_id=nid,
            description="Select option",
        ))

    elif role == "slider":
        ops.append(make_button_press(
            config=config, target_id=nid,
            description="Grab slider thumb",
        ))
        ops.append(GomsOperator(
            op_type=OperatorType.D,
            duration_s=0.90,
            target_id=nid,
            description="Drag slider",
        ))
        ops.append(make_button_release(config=config, target_id=nid))

    return ops, new_cursor


# ═══════════════════════════════════════════════════════════════════════════
# Method construction for UI patterns
# ═══════════════════════════════════════════════════════════════════════════

def build_method_for_pattern(
    pattern: UIPattern,
    tree: Any,
    goal_id: str,
    *,
    config: KLMConfig = KLMConfig(),
) -> GomsMethod:
    """Build a GOMS method for a recognized UI pattern.

    Parameters
    ----------
    pattern : UIPattern
        The recognized pattern.
    tree : Any
        Accessibility tree for node lookup.
    goal_id : str
        The goal this method achieves.
    config : KLMConfig
        Timing configuration.

    Returns
    -------
    GomsMethod
        A method with operators for this pattern.
    """
    method_id = f"method-{goal_id}-{pattern.pattern_type}"
    all_ops: List[GomsOperator] = []
    cursor = Point2D(0.0, 0.0)

    if pattern.pattern_type == "form_fill":
        all_ops.append(make_mental(
            config=config,
            description="Plan form filling",
        ))
        for nid in pattern.node_ids:
            node = tree.find_by_id(nid)
            if node is not None:
                ops, cursor = operators_for_node(node, cursor, config=config)
                all_ops.extend(ops)

    elif pattern.pattern_type == "navigation":
        all_ops.append(make_mental(
            config=config,
            n_choices=max(1, len(pattern.node_ids)),
            description=f"Choose from {len(pattern.node_ids)} nav items",
        ))
        # User picks one item (first as representative)
        if pattern.node_ids:
            node = tree.find_by_id(pattern.node_ids[0])
            if node is not None:
                ops, cursor = operators_for_node(node, cursor, config=config)
                all_ops.extend(ops)
            all_ops.append(make_system_response(
                config=config, description="Page navigation",
            ))

    elif pattern.pattern_type == "search_select":
        # Type in search, wait, then click result
        for nid in pattern.node_ids:
            node = tree.find_by_id(nid)
            if node is None:
                continue
            role = _node_role(node)
            if role == "textfield":
                ops, cursor = operators_for_node(node, cursor, config=config)
                all_ops.extend(ops)
                all_ops.append(make_system_response(
                    config=config, description="Search results appear",
                ))
                break
        # Click first result
        for nid in pattern.node_ids:
            node = tree.find_by_id(nid)
            if node is None:
                continue
            role = _node_role(node)
            if role in ("listitem", "link", "button"):
                ops, cursor = operators_for_node(node, cursor, config=config)
                all_ops.extend(ops)
                break

    elif pattern.pattern_type == "drag_drop":
        all_ops.append(make_mental(config=config, description="Plan drag"))
        if len(pattern.node_ids) >= 2:
            src_node = tree.find_by_id(pattern.node_ids[0])
            dst_node = tree.find_by_id(pattern.node_ids[1])
            if src_node is not None:
                src_bounds = _node_bounds(src_node)
                src_center = src_bounds.center if src_bounds else Point2D(100, 100)
                src_width = src_bounds.width if src_bounds else 60.0
                all_ops.append(make_pointing(
                    source=cursor, target_center=src_center,
                    target_width_px=src_width, config=config,
                    target_id=pattern.node_ids[0],
                ))
                all_ops.append(make_button_press(
                    config=config, target_id=pattern.node_ids[0],
                    description="Grab",
                ))
                cursor = src_center
            if dst_node is not None:
                dst_bounds = _node_bounds(dst_node)
                dst_center = dst_bounds.center if dst_bounds else Point2D(300, 100)
                dst_width = dst_bounds.width if dst_bounds else 60.0
                all_ops.append(GomsOperator(
                    op_type=OperatorType.D,
                    duration_s=0.90,
                    target_id=pattern.node_ids[1],
                    description="Drag to target",
                ))
                all_ops.append(make_button_release(
                    config=config, target_id=pattern.node_ids[1],
                    description="Drop",
                ))

    elif pattern.pattern_type == "wizard":
        all_ops.append(make_mental(
            config=config,
            description="Understand wizard step",
        ))
        for nid in pattern.node_ids:
            node = tree.find_by_id(nid)
            if node is not None:
                ops, cursor = operators_for_node(node, cursor, config=config)
                all_ops.extend(ops)
        all_ops.append(make_system_response(
            config=config, description="Wizard advances",
        ))

    else:
        # Default click pattern
        for nid in pattern.node_ids:
            node = tree.find_by_id(nid)
            if node is not None:
                ops, cursor = operators_for_node(node, cursor, config=config)
                all_ops.extend(ops)

    return GomsMethod(
        method_id=method_id,
        goal_id=goal_id,
        name=f"{pattern.pattern_type}_{pattern.description[:30]}",
        operators=tuple(all_ops),
        selection_rule=f"Use when pattern is {pattern.pattern_type}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical goal tree construction
# ═══════════════════════════════════════════════════════════════════════════

def build_goal_tree(
    task_description: str,
    patterns: Sequence[UIPattern],
    *,
    top_level_goal_id: Optional[str] = None,
) -> Tuple[List[GomsGoal], str]:
    """Build a hierarchical goal tree from recognized patterns.

    Creates a top-level goal and sub-goals for each pattern.

    Returns
    -------
    tuple[list[GomsGoal], str]
        List of all goals and the top-level goal ID.
    """
    top_id = top_level_goal_id or f"goal-{uuid.uuid4().hex[:8]}"
    goals: List[GomsGoal] = []
    subgoal_ids: List[str] = []

    for i, pattern in enumerate(patterns):
        sub_id = f"{top_id}-sub-{i}"
        subgoal_ids.append(sub_id)
        goals.append(GomsGoal(
            goal_id=sub_id,
            description=pattern.description or f"Complete {pattern.pattern_type}",
            parent_id=top_id,
        ))

    goals.insert(0, GomsGoal(
        goal_id=top_id,
        description=task_description,
        subgoal_ids=tuple(subgoal_ids),
    ))
    return goals, top_id


# ═══════════════════════════════════════════════════════════════════════════
# Goal conflict detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_goal_conflicts(goals: Sequence[GomsGoal], methods: Sequence[GomsMethod]) -> List[GoalConflict]:
    """Detect conflicts between goals.

    Conflict types:
    - **resource_conflict**: Two goals' methods target the same element.
    - **ordering_conflict**: Goal A's method produces system response
      that goal B depends on, but no ordering is specified.
    - **mutex_conflict**: Two goals modify the same state (e.g. both
      set the same radio group).

    Returns
    -------
    list[GoalConflict]
        Detected conflicts.
    """
    conflicts: List[GoalConflict] = []

    # Build goal → target_ids mapping
    goal_targets: Dict[str, Set[str]] = {}
    goal_has_response: Dict[str, bool] = {}

    for method in methods:
        targets: Set[str] = set()
        has_r = False
        for op in method.operators:
            if op.target_id:
                targets.add(op.target_id)
            if op.op_type == OperatorType.R:
                has_r = True
        goal_targets.setdefault(method.goal_id, set()).update(targets)
        goal_has_response[method.goal_id] = goal_has_response.get(
            method.goal_id, False
        ) or has_r

    # Check for resource conflicts (shared targets)
    goal_ids = list(goal_targets.keys())
    for i in range(len(goal_ids)):
        for j in range(i + 1, len(goal_ids)):
            shared = goal_targets[goal_ids[i]] & goal_targets[goal_ids[j]]
            if shared:
                conflicts.append(GoalConflict(
                    goal_a_id=goal_ids[i],
                    goal_b_id=goal_ids[j],
                    conflict_type="resource_conflict",
                    description=f"Shared targets: {', '.join(sorted(shared))}",
                ))

    # Check for ordering conflicts (response dependencies)
    for i in range(len(goal_ids)):
        for j in range(len(goal_ids)):
            if i == j:
                continue
            if goal_has_response.get(goal_ids[i], False):
                # Goal i produces a system response; if goal j
                # targets something that might depend on that response,
                # flag a potential ordering issue
                if goal_targets.get(goal_ids[j], set()):
                    # Only flag if j is a sibling (same parent)
                    gi = _find_goal(goals, goal_ids[i])
                    gj = _find_goal(goals, goal_ids[j])
                    if gi and gj and gi.parent_id == gj.parent_id:
                        conflicts.append(GoalConflict(
                            goal_a_id=goal_ids[i],
                            goal_b_id=goal_ids[j],
                            conflict_type="ordering_conflict",
                            description=(
                                f"Goal {goal_ids[i]} has system response; "
                                f"goal {goal_ids[j]} may depend on it"
                            ),
                        ))

    return conflicts


def _find_goal(goals: Sequence[GomsGoal], goal_id: str) -> Optional[GomsGoal]:
    for g in goals:
        if g.goal_id == goal_id:
            return g
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Full decomposition pipeline
# ═══════════════════════════════════════════════════════════════════════════

def decompose_task(
    tree: Any,
    task_description: str,
    *,
    config: KLMConfig = KLMConfig(),
    top_level_goal: Optional[str] = None,
) -> GomsModel:
    """Decompose a task into a full GOMS model via accessibility tree analysis.

    Pipeline:
    1. Recognize UI patterns from the tree.
    2. Build a hierarchical goal tree.
    3. Construct methods for each pattern.
    4. Detect conflicts.

    Returns
    -------
    GomsModel
        Complete GOMS model for the task.
    """
    patterns = recognize_patterns(tree)
    goals, top_id = build_goal_tree(
        task_description, patterns, top_level_goal_id=top_level_goal,
    )

    methods: List[GomsMethod] = []
    for i, pattern in enumerate(patterns):
        goal_id = goals[i + 1].goal_id  # skip top-level goal at index 0
        method = build_method_for_pattern(
            pattern, tree, goal_id, config=config,
        )
        methods.append(method)

    model_id = f"goms-{uuid.uuid4().hex[:8]}"
    return GomsModel(
        model_id=model_id,
        name=task_description,
        goals=tuple(goals),
        methods=tuple(methods),
        top_level_goal_id=top_id,
    )


__all__ = [
    "GoalConflict",
    "UIPattern",
    "build_goal_tree",
    "build_method_for_pattern",
    "decompose_task",
    "detect_goal_conflicts",
    "operators_for_node",
    "recognize_patterns",
]
