"""Unit tests for usability_oracle.wcag.evaluator — WCAG conformance evaluator.

Tests the WCAGConformanceEvaluator against mock accessibility trees
covering SC 1.1.1, 2.1.1, 2.4.1, 2.4.2, 2.4.6, and 4.1.2.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from usability_oracle.wcag.evaluator import WCAGConformanceEvaluator
from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
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


# ---------------------------------------------------------------------------
# Minimal well-formed trees
# ---------------------------------------------------------------------------

def _accessible_page() -> SimpleNamespace:
    """Minimal page that passes common checks."""
    skip_link = _node(role="link", name="Skip to main content", nid="skip",
                      properties={"tabindex": 0, "href": "#main"})
    heading = _node(role="heading", name="Welcome", nid="h1",
                    properties={"aria-level": 1})
    main = _node(role="main", name="", nid="main", children=[heading])
    root = _node(
        role="document", name="Test Page", nid="root",
        properties={"title": "Test Page", "lang": "en"},
        children=[skip_link, main],
    )
    return _tree(root, metadata={"url": "https://example.com"})


def _image_no_alt() -> SimpleNamespace:
    """Page with an image missing alt text."""
    img = _node(role="img", name="", nid="img1", properties={})
    root = _node(
        role="document", name="Images", nid="root",
        properties={"title": "Image Page", "lang": "en"},
        children=[_node(role="main", nid="main", children=[img])],
    )
    return _tree(root)


def _image_with_alt() -> SimpleNamespace:
    img = _node(role="img", name="Logo", nid="img1", properties={"alt": "Logo"})
    root = _node(
        role="document", name="Images", nid="root",
        properties={"title": "Image Page", "lang": "en"},
        children=[_node(role="main", nid="main", children=[img])],
    )
    return _tree(root)


def _keyboard_inaccessible() -> SimpleNamespace:
    """Interactive element with negative tabindex."""
    btn = _node(role="button", name="Submit", nid="btn1",
                properties={"tabindex": -1})
    root = _node(
        role="document", name="Form", nid="root",
        properties={"title": "Form Page", "lang": "en"},
        children=[_node(role="main", nid="main", children=[btn])],
    )
    return _tree(root)


def _onclick_no_keyboard() -> SimpleNamespace:
    """Custom div with onclick but no keyboard handler."""
    div = _node(role="generic", name="Click me", nid="div1",
                properties={"onclick": "doSomething()", "tabindex": 0})
    root = _node(
        role="document", name="Page", nid="root",
        properties={"title": "Page", "lang": "en"},
        children=[_node(role="main", nid="main", children=[div])],
    )
    return _tree(root)


def _no_title_page() -> SimpleNamespace:
    root = _node(role="generic", name="", nid="root", properties={},
                 children=[_node(role="main", nid="main")])
    return _tree(root)


def _no_skip_nav() -> SimpleNamespace:
    """Page with navigation but no skip link and no main landmark."""
    nav = _node(role="navigation", name="Nav", nid="nav1",
                children=[_node(role="link", name="Home", nid="l1")])
    root = _node(
        role="document", name="Page", nid="root",
        properties={"title": "Page", "lang": "en"},
        children=[nav, _node(role="generic", nid="content")],
    )
    return _tree(root)


def _unnamed_button() -> SimpleNamespace:
    """Button without an accessible name (violates 4.1.2)."""
    btn = _node(role="button", name="", nid="btn1")
    root = _node(
        role="document", name="Page", nid="root",
        properties={"title": "Page", "lang": "en"},
        children=[_node(role="main", nid="main", children=[btn])],
    )
    return _tree(root)


def _focus_trap_dialog() -> SimpleNamespace:
    """Modal dialog without close button — focus trap."""
    inner_btn = _node(role="button", name="OK", nid="ok")
    dialog = _node(role="dialog", name="Confirm", nid="dlg1",
                   properties={"aria-modal": "true"},
                   children=[inner_btn])
    root = _node(
        role="document", name="Page", nid="root",
        properties={"title": "Page", "lang": "en"},
        children=[_node(role="main", nid="main", children=[dialog])],
    )
    return _tree(root)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluator construction
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluatorConstruction:
    """Tests for creating an evaluator instance."""

    def test_default_construction(self) -> None:
        ev = WCAGConformanceEvaluator()
        assert ev is not None

    def test_supported_criteria_not_empty(self) -> None:
        ev = WCAGConformanceEvaluator()
        assert len(ev.supported_criteria) > 0
        assert "1.1.1" in ev.supported_criteria

    def test_custom_checker_registration(self) -> None:
        def _custom(tree: object) -> list:
            return []

        ev = WCAGConformanceEvaluator(custom_checkers={"99.9.9": _custom})
        assert "99.9.9" in ev.supported_criteria


# ═══════════════════════════════════════════════════════════════════════════
# SC 1.1.1 — Non-text Content
# ═══════════════════════════════════════════════════════════════════════════


class TestCheck111NonTextContent:
    """Tests for the 1.1.1 alt-text checker."""

    def test_image_without_alt_violates(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_image_no_alt(), "1.1.1")
        assert len(violations) >= 1
        assert all(v.sc_id == "1.1.1" for v in violations)

    def test_image_with_alt_passes(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_image_with_alt(), "1.1.1")
        assert len(violations) == 0

    def test_image_with_aria_label_passes(self) -> None:
        img = _node(role="img", name="", nid="img1",
                    properties={"aria-label": "Company logo"})
        root = _node(role="document", name="Page", nid="root",
                     properties={"title": "Page"},
                     children=[_node(role="main", nid="m", children=[img])])
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_tree(root), "1.1.1")
        assert len(violations) == 0

    def test_decorative_image_skipped(self) -> None:
        img = _node(role="presentation", name="", nid="deco1")
        root = _node(role="document", name="Page", nid="root",
                     properties={"title": "Page"},
                     children=[_node(role="main", nid="m", children=[img])])
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_tree(root), "1.1.1")
        assert len(violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SC 2.1.1 — Keyboard
# ═══════════════════════════════════════════════════════════════════════════


class TestCheck211Keyboard:
    """Tests for the 2.1.1 keyboard checker."""

    def test_negative_tabindex_on_button_violates(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_keyboard_inaccessible(), "2.1.1")
        assert any(v.sc_id == "2.1.1" for v in violations)

    def test_accessible_page_passes_keyboard(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_accessible_page(), "2.1.1")
        assert len(violations) == 0

    def test_focus_trap_detected(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_focus_trap_dialog(), "2.1.1")
        trap_violations = [v for v in violations if "focus trap" in v.message.lower()]
        assert len(trap_violations) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# SC 2.4.2 — Page Titled
# ═══════════════════════════════════════════════════════════════════════════


class TestCheck242PageTitled:
    """Tests for the 2.4.2 page-title checker."""

    def test_page_without_title_violates(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_no_title_page(), "2.4.2")
        assert any(v.sc_id == "2.4.2" for v in violations)

    def test_page_with_title_passes(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_accessible_page(), "2.4.2")
        assert len(violations) == 0

    def test_document_role_with_name_passes(self) -> None:
        root = _node(role="document", name="My Document", nid="root",
                     properties={}, children=[_node(role="main", nid="m")])
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_tree(root), "2.4.2")
        assert len(violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SC 2.4.1 — Bypass Blocks
# ═══════════════════════════════════════════════════════════════════════════


class TestCheck241BypassBlocks:
    """Tests for the 2.4.1 skip navigation checker."""

    def test_no_skip_nav_and_no_main_violates(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_no_skip_nav(), "2.4.1")
        assert len(violations) >= 1

    def test_main_landmark_satisfies(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_accessible_page(), "2.4.1")
        assert len(violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SC 4.1.2 — Name, Role, Value
# ═══════════════════════════════════════════════════════════════════════════


class TestCheck412NameRoleValue:
    """Tests for the 4.1.2 ARIA roles / name checker."""

    def test_unnamed_button_violates(self) -> None:
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_unnamed_button(), "4.1.2")
        assert any(v.sc_id == "4.1.2" for v in violations)

    def test_named_button_passes(self) -> None:
        btn = _node(role="button", name="OK", nid="btn1")
        root = _node(role="document", name="Page", nid="root",
                     properties={"title": "Page"},
                     children=[_node(role="main", nid="m", children=[btn])])
        ev = WCAGConformanceEvaluator()
        violations = ev.check_criterion(_tree(root), "4.1.2")
        btn_violations = [v for v in violations if v.node_id == "btn1"]
        assert len(btn_violations) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Full evaluation
# ═══════════════════════════════════════════════════════════════════════════


class TestFullEvaluation:
    """Tests for evaluate() returning WCAGResult."""

    def test_evaluate_returns_wcag_result(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), ConformanceLevel.A)
        assert isinstance(result, WCAGResult)

    def test_evaluate_records_criteria_tested(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), ConformanceLevel.A)
        assert result.criteria_tested > 0

    def test_evaluate_level_string(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), "A")
        assert isinstance(result, WCAGResult)
        assert result.target_level == ConformanceLevel.A

    def test_evaluate_criteria_ids_filter(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), ConformanceLevel.A,
                             criteria_ids=["1.1.1"])
        assert result.criteria_tested <= 1

    def test_evaluate_page_url_from_metadata(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), ConformanceLevel.A)
        assert result.page_url == "https://example.com"

    def test_violations_on_bad_page(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_image_no_alt(), ConformanceLevel.A)
        assert result.violation_count > 0


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregation:
    """Tests for violations_by_level / principle / criterion."""

    def test_violations_by_level(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_image_no_alt(), ConformanceLevel.AA)
        by_level = ev.violations_by_level(result)
        assert isinstance(by_level, dict)
        assert ConformanceLevel.A in by_level

    def test_violations_by_principle(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_image_no_alt(), ConformanceLevel.AA)
        by_principle = ev.violations_by_principle(result)
        assert isinstance(by_principle, dict)
        assert WCAGPrinciple.PERCEIVABLE in by_principle

    def test_violations_by_criterion(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_image_no_alt(), ConformanceLevel.A)
        by_sc = ev.violations_by_criterion(result)
        assert isinstance(by_sc, dict)
        if result.violation_count > 0:
            assert any(sc_id.startswith("1.") for sc_id in by_sc)

    def test_conformance_ratio(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), ConformanceLevel.A)
        assert 0.0 <= result.conformance_ratio <= 1.0

    def test_is_conformant_on_clean_page(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_accessible_page(), ConformanceLevel.A)
        assert result.is_conformant is True

    def test_wcag_result_violations_by_impact(self) -> None:
        ev = WCAGConformanceEvaluator()
        result = ev.evaluate(_image_no_alt(), ConformanceLevel.A)
        by_impact = result.violations_by_impact
        assert ImpactLevel.CRITICAL in by_impact
