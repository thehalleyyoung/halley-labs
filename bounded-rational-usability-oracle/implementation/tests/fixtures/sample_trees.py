"""Sample accessibility trees for testing.

Provides factory functions that build realistic AccessibilityNode / AccessibilityTree
objects representing common UI patterns.
"""

from __future__ import annotations

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)


def _default_state(**overrides) -> AccessibilityState:
    defaults = dict(
        focused=False, selected=False, expanded=False, checked=None,
        disabled=False, hidden=False, required=False, readonly=False,
        pressed=None, value=None,
    )
    defaults.update(overrides)
    return AccessibilityState(**defaults)


def make_simple_form_tree() -> AccessibilityTree:
    """A login form with username, password, and submit button."""
    submit_btn = AccessibilityNode(
        id="btn_submit", role="button", name="Submit",
        description="", bounding_box=BoundingBox(x=50, y=200, width=100, height=40),
        properties={}, state=_default_state(), children=[], parent_id="form1",
        depth=2, index_in_parent=2,
    )
    pw_input = AccessibilityNode(
        id="input_pw", role="textbox", name="Password",
        description="Enter your password",
        bounding_box=BoundingBox(x=50, y=130, width=200, height=30),
        properties={"type": "password"}, state=_default_state(required=True),
        children=[], parent_id="form1", depth=2, index_in_parent=1,
    )
    user_input = AccessibilityNode(
        id="input_user", role="textbox", name="Username",
        description="Enter your username",
        bounding_box=BoundingBox(x=50, y=60, width=200, height=30),
        properties={}, state=_default_state(required=True),
        children=[], parent_id="form1", depth=2, index_in_parent=0,
    )
    form_node = AccessibilityNode(
        id="form1", role="form", name="Login",
        description="Login form",
        bounding_box=BoundingBox(x=20, y=20, width=300, height=260),
        properties={}, state=_default_state(),
        children=[user_input, pw_input, submit_btn],
        parent_id="root", depth=1, index_in_parent=0,
    )
    root = AccessibilityNode(
        id="root", role="document", name="Login Page",
        description="",
        bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
        properties={}, state=_default_state(),
        children=[form_node], parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def make_navigation_tree() -> AccessibilityTree:
    """A navigation menu with 5 links."""
    links = []
    for i, label in enumerate(["Home", "Products", "About", "Contact", "Help"]):
        links.append(AccessibilityNode(
            id=f"link_{i}", role="link", name=label, description="",
            bounding_box=BoundingBox(x=i * 120, y=0, width=100, height=40),
            properties={"href": f"/{label.lower()}"},
            state=_default_state(), children=[], parent_id="nav",
            depth=2, index_in_parent=i,
        ))
    nav = AccessibilityNode(
        id="nav", role="navigation", name="Main Navigation", description="",
        bounding_box=BoundingBox(x=0, y=0, width=600, height=40),
        properties={}, state=_default_state(), children=links,
        parent_id="root", depth=1, index_in_parent=0,
    )
    root = AccessibilityNode(
        id="root", role="document", name="Site", description="",
        bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
        properties={}, state=_default_state(), children=[nav],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def make_dashboard_tree() -> AccessibilityTree:
    """A complex dashboard with multiple regions, tables, and controls."""
    cells = []
    for r in range(3):
        for c in range(4):
            cells.append(AccessibilityNode(
                id=f"cell_{r}_{c}", role="cell", name=f"Value {r},{c}",
                description="",
                bounding_box=BoundingBox(x=100 + c * 80, y=200 + r * 30, width=70, height=25),
                properties={}, state=_default_state(), children=[],
                parent_id="table1", depth=3, index_in_parent=r * 4 + c,
            ))
    table = AccessibilityNode(
        id="table1", role="table", name="Metrics", description="",
        bounding_box=BoundingBox(x=100, y=200, width=320, height=90),
        properties={}, state=_default_state(), children=cells,
        parent_id="main_region", depth=2, index_in_parent=0,
    )
    chart = AccessibilityNode(
        id="chart1", role="img", name="Revenue Chart", description="Bar chart of revenue",
        bounding_box=BoundingBox(x=500, y=200, width=400, height=300),
        properties={}, state=_default_state(), children=[],
        parent_id="main_region", depth=2, index_in_parent=1,
    )
    main_region = AccessibilityNode(
        id="main_region", role="main", name="Dashboard", description="",
        bounding_box=BoundingBox(x=0, y=60, width=1920, height=1020),
        properties={}, state=_default_state(), children=[table, chart],
        parent_id="root", depth=1, index_in_parent=0,
    )
    root = AccessibilityNode(
        id="root", role="document", name="Dashboard", description="",
        bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
        properties={}, state=_default_state(), children=[main_region],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def make_modal_dialog_tree() -> AccessibilityTree:
    """A modal dialog overlaying a page."""
    ok_btn = AccessibilityNode(
        id="btn_ok", role="button", name="OK", description="",
        bounding_box=BoundingBox(x=700, y=500, width=80, height=36),
        properties={}, state=_default_state(), children=[],
        parent_id="dialog1", depth=2, index_in_parent=1,
    )
    cancel_btn = AccessibilityNode(
        id="btn_cancel", role="button", name="Cancel", description="",
        bounding_box=BoundingBox(x=800, y=500, width=80, height=36),
        properties={}, state=_default_state(), children=[],
        parent_id="dialog1", depth=2, index_in_parent=2,
    )
    message = AccessibilityNode(
        id="msg1", role="heading", name="Confirm Deletion", description="",
        bounding_box=BoundingBox(x=600, y=400, width=300, height=30),
        properties={"level": 2}, state=_default_state(), children=[],
        parent_id="dialog1", depth=2, index_in_parent=0,
    )
    dialog = AccessibilityNode(
        id="dialog1", role="dialog", name="Confirm", description="",
        bounding_box=BoundingBox(x=560, y=340, width=400, height=250),
        properties={"modal": True}, state=_default_state(),
        children=[message, ok_btn, cancel_btn],
        parent_id="root", depth=1, index_in_parent=0,
    )
    root = AccessibilityNode(
        id="root", role="document", name="App", description="",
        bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
        properties={}, state=_default_state(), children=[dialog],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})
