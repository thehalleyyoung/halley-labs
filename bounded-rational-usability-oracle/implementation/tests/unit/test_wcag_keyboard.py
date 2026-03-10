"""Unit tests for usability_oracle.wcag.keyboard — keyboard accessibility.

Tests tab-order extraction, focus-trap detection, shortcut-conflict
detection, skip-navigation verification, focus indicators, and
navigation cost estimation using mock accessibility trees.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from usability_oracle.wcag.keyboard import (
    FocusIndicatorIssue,
    FocusTrap,
    FocusableElement,
    NavigationCost,
    ShortcutConflict,
    SkipNavResult,
    check_focus_indicators,
    compute_navigation_cost,
    detect_focus_traps,
    detect_shortcut_conflicts,
    extract_tab_order,
    verify_skip_navigation,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _node(
    role: str = "",
    name: str = "",
    nid: str = "",
    properties: dict | None = None,
    children: list | None = None,
    depth: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=nid,
        node_id=nid,
        role=role,
        name=name,
        properties=properties or {},
        children=children or [],
        depth=depth,
        bounding_box=None,
        state=SimpleNamespace(hidden=False, disabled=False),
    )


def _tree(root: SimpleNamespace, metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(root=root, metadata=metadata or {})


# ═══════════════════════════════════════════════════════════════════════════
# extract_tab_order
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractTabOrder:
    """Tests for tab-order computation."""

    def test_positive_tabindex_comes_first(self) -> None:
        btn_tab2 = _node(role="button", name="Second", nid="b2",
                         properties={"tabindex": 2})
        btn_tab1 = _node(role="button", name="First", nid="b1",
                         properties={"tabindex": 1})
        btn_tab0 = _node(role="button", name="Third", nid="b3",
                         properties={"tabindex": 0})
        root = _node(children=[btn_tab2, btn_tab1, btn_tab0])
        order = extract_tab_order(_tree(root))
        ids = [e.node_id for e in order]
        assert ids.index("b1") < ids.index("b3")
        assert ids.index("b2") < ids.index("b3")

    def test_negative_tabindex_excluded(self) -> None:
        btn = _node(role="button", name="Hidden", nid="b1",
                    properties={"tabindex": -1})
        root = _node(children=[btn])
        order = extract_tab_order(_tree(root))
        ids = [e.node_id for e in order]
        assert "b1" not in ids

    def test_zero_tabindex_in_dom_order(self) -> None:
        b1 = _node(role="button", name="A", nid="a", properties={"tabindex": 0})
        b2 = _node(role="button", name="B", nid="b", properties={"tabindex": 0})
        root = _node(children=[b1, b2])
        order = extract_tab_order(_tree(root))
        ids = [e.node_id for e in order]
        assert ids.index("a") < ids.index("b")

    def test_natively_focusable_included_without_tabindex(self) -> None:
        link = _node(role="link", name="Go", nid="lnk")
        root = _node(children=[link])
        order = extract_tab_order(_tree(root))
        assert any(e.node_id == "lnk" for e in order)

    def test_hidden_node_excluded(self) -> None:
        btn = _node(role="button", name="Hidden", nid="h1")
        btn.state.hidden = True
        root = _node(children=[btn])
        order = extract_tab_order(_tree(root))
        assert all(e.node_id != "h1" for e in order)

    def test_disabled_node_excluded(self) -> None:
        btn = _node(role="button", name="Disabled", nid="d1")
        btn.state.disabled = True
        root = _node(children=[btn])
        order = extract_tab_order(_tree(root))
        assert all(e.node_id != "d1" for e in order)

    def test_empty_tree(self) -> None:
        root = _node()
        order = extract_tab_order(_tree(root))
        assert len(order) == 0

    def test_focusable_element_attributes(self) -> None:
        btn = _node(role="button", name="OK", nid="ok", properties={"tabindex": 0})
        root = _node(children=[btn])
        order = extract_tab_order(_tree(root))
        assert len(order) == 1
        elem = order[0]
        assert isinstance(elem, FocusableElement)
        assert elem.role == "button"
        assert elem.name == "OK"


# ═══════════════════════════════════════════════════════════════════════════
# detect_focus_traps
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectFocusTraps:
    """Tests for keyboard focus-trap detection."""

    def test_modal_without_close_is_trap(self) -> None:
        inner = _node(role="button", name="OK", nid="ok")
        dialog = _node(role="dialog", name="Confirm", nid="dlg",
                       properties={"aria-modal": "true"},
                       children=[inner])
        root = _node(children=[dialog])
        traps = detect_focus_traps(_tree(root))
        assert len(traps) >= 1
        assert isinstance(traps[0], FocusTrap)
        assert traps[0].container_id == "dlg"

    def test_modal_with_close_is_not_trap(self) -> None:
        close_btn = _node(role="button", name="Close", nid="close")
        inner = _node(role="button", name="OK", nid="ok")
        dialog = _node(role="dialog", name="Confirm", nid="dlg",
                       properties={"aria-modal": "true"},
                       children=[inner, close_btn])
        root = _node(children=[dialog])
        traps = detect_focus_traps(_tree(root))
        assert len(traps) == 0

    def test_non_modal_dialog_not_trapped(self) -> None:
        inner = _node(role="button", name="OK", nid="ok")
        dialog = _node(role="dialog", name="Info", nid="dlg",
                       properties={"aria-modal": "false"},
                       children=[inner])
        root = _node(children=[dialog])
        traps = detect_focus_traps(_tree(root))
        assert len(traps) == 0

    def test_alertdialog_modal_trapped(self) -> None:
        inner = _node(role="button", name="Retry", nid="retry")
        dialog = _node(role="alertdialog", name="Error", nid="adlg",
                       properties={"aria-modal": "true"},
                       children=[inner])
        root = _node(children=[dialog])
        traps = detect_focus_traps(_tree(root))
        assert len(traps) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# detect_shortcut_conflicts
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectShortcutConflicts:
    """Tests for keyboard shortcut conflict detection."""

    def test_single_char_shortcut_flagged(self) -> None:
        btn = _node(role="button", name="Bold", nid="b1",
                    properties={"aria-keyshortcuts": "b"})
        root = _node(children=[btn])
        conflicts = detect_shortcut_conflicts(_tree(root))
        single = [c for c in conflicts if len(c.key) == 1]
        assert len(single) >= 1

    def test_duplicate_binding_flagged(self) -> None:
        b1 = _node(role="button", name="Bold", nid="b1",
                   properties={"accesskey": "b"})
        b2 = _node(role="button", name="Back", nid="b2",
                   properties={"accesskey": "b"})
        root = _node(children=[b1, b2])
        conflicts = detect_shortcut_conflicts(_tree(root))
        dup = [c for c in conflicts if len(c.node_ids) > 1]
        assert len(dup) >= 1

    def test_no_conflicts_clean_tree(self) -> None:
        btn = _node(role="button", name="OK", nid="ok")
        root = _node(children=[btn])
        conflicts = detect_shortcut_conflicts(_tree(root))
        assert len(conflicts) == 0

    def test_modifier_key_shortcut_not_flagged_as_single(self) -> None:
        btn = _node(role="button", name="Save", nid="s1",
                    properties={"aria-keyshortcuts": "Control+s"})
        root = _node(children=[btn])
        conflicts = detect_shortcut_conflicts(_tree(root))
        single = [c for c in conflicts if c.reason.startswith("Single character")]
        assert len(single) == 0


# ═══════════════════════════════════════════════════════════════════════════
# verify_skip_navigation
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifySkipNavigation:
    """Tests for skip-navigation verification (SC 2.4.1)."""

    def test_main_landmark_satisfies(self) -> None:
        main = _node(role="main", name="", nid="main")
        root = _node(children=[main])
        result = verify_skip_navigation(_tree(root))
        assert isinstance(result, SkipNavResult)
        assert result.main_landmark_present is True
        assert len(result.violations) == 0

    def test_no_skip_and_no_main_violates(self) -> None:
        nav = _node(role="navigation", name="Nav", nid="nav")
        root = _node(children=[nav, _node(role="generic", nid="c")])
        result = verify_skip_navigation(_tree(root))
        assert len(result.violations) >= 1

    def test_skip_link_detected(self) -> None:
        skip = _node(role="link", name="Skip to main content", nid="skip",
                     properties={"tabindex": 0})
        root = _node(children=[skip, _node(role="generic", nid="c")])
        result = verify_skip_navigation(_tree(root))
        assert result.has_skip_link is True

    def test_navigation_landmark_count(self) -> None:
        nav1 = _node(role="navigation", name="Primary", nid="n1")
        nav2 = _node(role="navigation", name="Footer", nid="n2")
        main = _node(role="main", nid="main")
        root = _node(children=[nav1, main, nav2])
        result = verify_skip_navigation(_tree(root))
        assert result.navigation_landmarks == 2


# ═══════════════════════════════════════════════════════════════════════════
# compute_navigation_cost
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeNavigationCost:
    """Tests for navigation cost estimation."""

    def test_empty_tree_zero_cost(self) -> None:
        root = _node()
        cost = compute_navigation_cost(_tree(root))
        assert isinstance(cost, NavigationCost)
        assert cost.total_focusable == 0
        assert cost.estimated_time_seconds == 0.0

    def test_single_element(self) -> None:
        btn = _node(role="button", name="OK", nid="ok", properties={"tabindex": 0})
        root = _node(children=[btn])
        cost = compute_navigation_cost(_tree(root))
        assert cost.total_focusable == 1
        assert cost.max_tabs_to_target == 1

    def test_target_reduces_cost(self) -> None:
        buttons = [
            _node(role="button", name=f"B{i}", nid=f"b{i}", properties={"tabindex": 0})
            for i in range(10)
        ]
        root = _node(children=buttons)
        cost_no_target = compute_navigation_cost(_tree(root))
        cost_first = compute_navigation_cost(_tree(root), target_id="b0")
        assert cost_first.estimated_time_seconds <= cost_no_target.estimated_time_seconds

    def test_entropy_positive_for_mixed_roles(self) -> None:
        nodes = [
            _node(role="button", name="B", nid="b1", properties={"tabindex": 0}),
            _node(role="link", name="L", nid="l1", properties={"tabindex": 0}),
            _node(role="textbox", name="T", nid="t1", properties={"tabindex": 0}),
        ]
        root = _node(children=nodes)
        cost = compute_navigation_cost(_tree(root))
        assert cost.tab_order_entropy > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# check_focus_indicators
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckFocusIndicators:
    """Tests for focus-indicator heuristic checks."""

    def test_outline_none_flagged(self) -> None:
        btn = _node(role="button", name="OK", nid="btn1",
                    properties={"style": "outline: none;"})
        root = _node(children=[btn])
        issues = check_focus_indicators(_tree(root))
        assert any(isinstance(i, FocusIndicatorIssue) for i in issues)

    def test_no_issues_clean_button(self) -> None:
        btn = _node(role="button", name="OK", nid="btn1", properties={})
        root = _node(children=[btn])
        issues = check_focus_indicators(_tree(root))
        assert len(issues) == 0

    def test_small_bounding_box_flagged(self) -> None:
        btn = _node(role="button", name="X", nid="btn1", properties={})
        btn.bounding_box = SimpleNamespace(x=0, y=0, width=5, height=5)
        root = _node(children=[btn])
        issues = check_focus_indicators(_tree(root))
        assert any("too small" in i.reason for i in issues)
