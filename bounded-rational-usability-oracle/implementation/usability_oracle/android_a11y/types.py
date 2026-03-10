"""
usability_oracle.android_a11y.types — Data types for Android accessibility format.

Provides immutable value types for representing Android view hierarchies,
accessibility node info, content descriptions, and accessibility actions
as defined by the Android Accessibility API (``AccessibilityNodeInfo``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntFlag, unique
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from usability_oracle.core.types import BoundingBox, Point2D


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class AndroidClassName(Enum):
    """Common Android widget class names relevant to accessibility."""

    VIEW = "android.view.View"
    TEXT_VIEW = "android.widget.TextView"
    IMAGE_VIEW = "android.widget.ImageView"
    BUTTON = "android.widget.Button"
    IMAGE_BUTTON = "android.widget.ImageButton"
    EDIT_TEXT = "android.widget.EditText"
    CHECK_BOX = "android.widget.CheckBox"
    RADIO_BUTTON = "android.widget.RadioButton"
    TOGGLE_BUTTON = "android.widget.ToggleButton"
    SWITCH = "android.widget.Switch"
    SEEK_BAR = "android.widget.SeekBar"
    SPINNER = "android.widget.Spinner"
    PROGRESS_BAR = "android.widget.ProgressBar"
    SCROLL_VIEW = "android.widget.ScrollView"
    RECYCLER_VIEW = "androidx.recyclerview.widget.RecyclerView"
    WEB_VIEW = "android.webkit.WebView"
    VIEW_GROUP = "android.view.ViewGroup"
    LINEAR_LAYOUT = "android.widget.LinearLayout"
    RELATIVE_LAYOUT = "android.widget.RelativeLayout"
    FRAME_LAYOUT = "android.widget.FrameLayout"
    TAB_WIDGET = "android.widget.TabWidget"


@unique
class AccessibilityActionId(Enum):
    """Standard Android accessibility action identifiers."""

    CLICK = "ACTION_CLICK"
    LONG_CLICK = "ACTION_LONG_CLICK"
    FOCUS = "ACTION_ACCESSIBILITY_FOCUS"
    CLEAR_FOCUS = "ACTION_CLEAR_ACCESSIBILITY_FOCUS"
    SELECT = "ACTION_SELECT"
    CLEAR_SELECTION = "ACTION_CLEAR_SELECTION"
    SCROLL_FORWARD = "ACTION_SCROLL_FORWARD"
    SCROLL_BACKWARD = "ACTION_SCROLL_BACKWARD"
    NEXT_AT_MOVEMENT_GRANULARITY = "ACTION_NEXT_AT_MOVEMENT_GRANULARITY"
    PREVIOUS_AT_MOVEMENT_GRANULARITY = "ACTION_PREVIOUS_AT_MOVEMENT_GRANULARITY"
    SET_TEXT = "ACTION_SET_TEXT"
    COPY = "ACTION_COPY"
    PASTE = "ACTION_PASTE"
    CUT = "ACTION_CUT"
    SET_SELECTION = "ACTION_SET_SELECTION"
    EXPAND = "ACTION_EXPAND"
    COLLAPSE = "ACTION_COLLAPSE"
    DISMISS = "ACTION_DISMISS"
    SHOW_ON_SCREEN = "ACTION_SHOW_ON_SCREEN"


# ═══════════════════════════════════════════════════════════════════════════
# BoundsInfo
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BoundsInfo:
    """Bounding-box coordinates in both screen and parent-relative frames.

    Android accessibility nodes report two coordinate systems:
    ``boundsInScreen`` (absolute) and ``boundsInParent`` (relative to
    the parent's top-left corner).

    Attributes:
        screen_left: Left edge in screen coordinates (px).
        screen_top: Top edge in screen coordinates (px).
        screen_right: Right edge in screen coordinates (px).
        screen_bottom: Bottom edge in screen coordinates (px).
        parent_left: Left edge relative to parent (px).
        parent_top: Top edge relative to parent (px).
        parent_right: Right edge relative to parent (px).
        parent_bottom: Bottom edge relative to parent (px).
    """

    screen_left: int
    screen_top: int
    screen_right: int
    screen_bottom: int
    parent_left: int
    parent_top: int
    parent_right: int
    parent_bottom: int

    @property
    def screen_width(self) -> int:
        """Width in screen coordinates."""
        return self.screen_right - self.screen_left

    @property
    def screen_height(self) -> int:
        """Height in screen coordinates."""
        return self.screen_bottom - self.screen_top

    @property
    def screen_center(self) -> Point2D:
        """Centre point in screen coordinates."""
        return Point2D(
            (self.screen_left + self.screen_right) / 2.0,
            (self.screen_top + self.screen_bottom) / 2.0,
        )

    def to_bounding_box(self) -> BoundingBox:
        """Convert screen bounds to a core :class:`BoundingBox`."""
        return BoundingBox(
            x=float(self.screen_left),
            y=float(self.screen_top),
            width=float(self.screen_width),
            height=float(self.screen_height),
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "screen_left": self.screen_left,
            "screen_top": self.screen_top,
            "screen_right": self.screen_right,
            "screen_bottom": self.screen_bottom,
            "parent_left": self.parent_left,
            "parent_top": self.parent_top,
            "parent_right": self.parent_right,
            "parent_bottom": self.parent_bottom,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BoundsInfo:
        return cls(
            screen_left=int(d["screen_left"]),
            screen_top=int(d["screen_top"]),
            screen_right=int(d["screen_right"]),
            screen_bottom=int(d["screen_bottom"]),
            parent_left=int(d["parent_left"]),
            parent_top=int(d["parent_top"]),
            parent_right=int(d["parent_right"]),
            parent_bottom=int(d["parent_bottom"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ContentDescription
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ContentDescription:
    """Accessibility label / content description for an Android node.

    Attributes:
        text: The visible text of the node (``getText()``).
        content_description: Explicit content description
            (``getContentDescription()``).
        hint_text: Hint text for editable fields (``getHintText()``).
        tooltip_text: Tooltip text (``getTooltipText()``).
        label_for: Resource id of the node this one labels.
        labeled_by: Resource id of the node that labels this one.
    """

    text: Optional[str] = None
    content_description: Optional[str] = None
    hint_text: Optional[str] = None
    tooltip_text: Optional[str] = None
    label_for: Optional[str] = None
    labeled_by: Optional[str] = None

    @property
    def accessible_name(self) -> Optional[str]:
        """Computed accessible name following Android precedence rules.

        Precedence: ``contentDescription`` → ``text`` → ``hintText``.
        """
        return self.content_description or self.text or self.hint_text

    @property
    def has_label(self) -> bool:
        """Whether any form of accessible name is present."""
        return self.accessible_name is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "content_description": self.content_description,
            "hint_text": self.hint_text,
            "tooltip_text": self.tooltip_text,
            "label_for": self.label_for,
            "labeled_by": self.labeled_by,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ContentDescription:
        return cls(
            text=d.get("text"),
            content_description=d.get("content_description"),
            hint_text=d.get("hint_text"),
            tooltip_text=d.get("tooltip_text"),
            label_for=d.get("label_for"),
            labeled_by=d.get("labeled_by"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# AccessibilityAction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AccessibilityAction:
    """An accessibility action available on an Android node.

    Attributes:
        action_id: Standard or custom action identifier.
        label: Human-readable label for the action (may be ``None``
            for standard actions).
    """

    action_id: str
    label: Optional[str] = None

    @property
    def is_standard(self) -> bool:
        """Whether this is a standard (non-custom) action."""
        return self.action_id.startswith("ACTION_")

    def to_dict(self) -> Dict[str, Any]:
        return {"action_id": self.action_id, "label": self.label}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AccessibilityAction:
        return cls(
            action_id=str(d["action_id"]),
            label=d.get("label"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# AndroidNode
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AndroidNode:
    """A single node in an Android view accessibility hierarchy.

    Models the information exposed by ``AccessibilityNodeInfo`` in the
    Android framework.

    Attributes:
        node_id: Unique identifier (resource-id or generated).
        class_name: Fully-qualified Android widget class name.
        package_name: Application package name.
        bounds: Bounding-box information (screen and parent coordinates).
        description: Content description, text, and labelling info.
        actions: Available accessibility actions.
        is_clickable: Whether the node responds to click events.
        is_focusable: Whether the node can receive accessibility focus.
        is_focused: Whether the node currently has accessibility focus.
        is_scrollable: Whether the node supports scrolling.
        is_checkable: Whether the node is checkable (checkbox, switch).
        is_checked: Whether the node is currently checked.
        is_enabled: Whether the node is enabled.
        is_visible_to_user: Whether the node is visible on screen.
        is_important_for_accessibility: Android's importance flag.
        child_ids: Ordered list of child node identifiers.
        depth: Depth in the view hierarchy (root = 0).
    """

    node_id: str
    class_name: str
    package_name: str
    bounds: BoundsInfo
    description: ContentDescription
    actions: Tuple[AccessibilityAction, ...] = ()
    is_clickable: bool = False
    is_focusable: bool = False
    is_focused: bool = False
    is_scrollable: bool = False
    is_checkable: bool = False
    is_checked: bool = False
    is_enabled: bool = True
    is_visible_to_user: bool = True
    is_important_for_accessibility: bool = True
    child_ids: Tuple[str, ...] = ()
    depth: int = 0

    @property
    def is_interactive(self) -> bool:
        """Whether the node is an interactive widget."""
        return self.is_clickable or self.is_focusable or self.is_checkable

    @property
    def accessible_name(self) -> Optional[str]:
        """Computed accessible name (delegated to ContentDescription)."""
        return self.description.accessible_name

    @property
    def num_children(self) -> int:
        """Number of direct children."""
        return len(self.child_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "class_name": self.class_name,
            "package_name": self.package_name,
            "bounds": self.bounds.to_dict(),
            "description": self.description.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "is_clickable": self.is_clickable,
            "is_focusable": self.is_focusable,
            "is_focused": self.is_focused,
            "is_scrollable": self.is_scrollable,
            "is_checkable": self.is_checkable,
            "is_checked": self.is_checked,
            "is_enabled": self.is_enabled,
            "is_visible_to_user": self.is_visible_to_user,
            "is_important_for_accessibility": self.is_important_for_accessibility,
            "child_ids": list(self.child_ids),
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AndroidNode:
        return cls(
            node_id=str(d["node_id"]),
            class_name=str(d["class_name"]),
            package_name=str(d["package_name"]),
            bounds=BoundsInfo.from_dict(d["bounds"]),
            description=ContentDescription.from_dict(d["description"]),
            actions=tuple(
                AccessibilityAction.from_dict(a) for a in d.get("actions", [])
            ),
            is_clickable=bool(d.get("is_clickable", False)),
            is_focusable=bool(d.get("is_focusable", False)),
            is_focused=bool(d.get("is_focused", False)),
            is_scrollable=bool(d.get("is_scrollable", False)),
            is_checkable=bool(d.get("is_checkable", False)),
            is_checked=bool(d.get("is_checked", False)),
            is_enabled=bool(d.get("is_enabled", True)),
            is_visible_to_user=bool(d.get("is_visible_to_user", True)),
            is_important_for_accessibility=bool(
                d.get("is_important_for_accessibility", True)
            ),
            child_ids=tuple(d.get("child_ids", [])),
            depth=int(d.get("depth", 0)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ViewHierarchy
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ViewHierarchy:
    """Complete Android view accessibility hierarchy.

    Represents the full tree dumped by ``uiautomator dump`` or the
    Accessibility Service, rooted at the active window.

    Attributes:
        root_id: Node id of the root of the hierarchy.
        nodes: Mapping from node id to :class:`AndroidNode`.
        package_name: Application package name.
        window_title: Title of the active window.
        screen_width: Device screen width in pixels.
        screen_height: Device screen height in pixels.
        api_level: Android API level of the device.
    """

    root_id: str
    nodes: Dict[str, AndroidNode]
    package_name: str
    window_title: str = ""
    screen_width: int = 1080
    screen_height: int = 1920
    api_level: int = 33

    @property
    def root(self) -> AndroidNode:
        """The root node of the hierarchy."""
        return self.nodes[self.root_id]

    @property
    def node_count(self) -> int:
        """Total number of nodes in the hierarchy."""
        return len(self.nodes)

    @property
    def interactive_nodes(self) -> Sequence[AndroidNode]:
        """All nodes that are interactive (clickable/focusable/checkable)."""
        return [n for n in self.nodes.values() if n.is_interactive]

    @property
    def max_depth(self) -> int:
        """Maximum depth across all nodes."""
        return max((n.depth for n in self.nodes.values()), default=0)

    def get_node(self, node_id: str) -> Optional[AndroidNode]:
        """Look up a node by id."""
        return self.nodes.get(node_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "package_name": self.package_name,
            "window_title": self.window_title,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "api_level": self.api_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ViewHierarchy:
        return cls(
            root_id=str(d["root_id"]),
            nodes={k: AndroidNode.from_dict(v) for k, v in d["nodes"].items()},
            package_name=str(d["package_name"]),
            window_title=str(d.get("window_title", "")),
            screen_width=int(d.get("screen_width", 1080)),
            screen_height=int(d.get("screen_height", 1920)),
            api_level=int(d.get("api_level", 33)),
        )
