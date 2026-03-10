"""Tests for usability_oracle.accessibility.json_parser — JSON to AccessibilityTree.

Covers the JSONAccessibilityParser with parse() for JSON strings,
parse_dict() for dict input, format detection (chrome_devtools, generic),
fixture loading, and error handling for invalid/malformed input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usability_oracle.accessibility.json_parser import JSONAccessibilityParser, _detect_format, _Format
from usability_oracle.accessibility.models import AccessibilityTree

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_json"


def _load_json(name: str) -> str:
    """Load a sample JSON fixture by filename as a raw string."""
    return (_FIXTURES / name).read_text()


def _load_json_dict(name: str) -> dict:
    """Load a sample JSON fixture by filename as a parsed dict."""
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture
def parser() -> JSONAccessibilityParser:
    """Return a default JSONAccessibilityParser instance."""
    return JSONAccessibilityParser()


# ── Simple form JSON ──────────────────────────────────────────────────────────


class TestSimpleFormJSON:
    """Tests for parsing the simple_form.json fixture."""

    def test_parse_returns_tree(self, parser):
        """parse() with a JSON string should return an AccessibilityTree."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        assert isinstance(tree, AccessibilityTree)

    def test_root_is_document(self, parser):
        """The root of the simple form tree should have role 'document'."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        assert tree.root.role == "document"

    def test_contains_form(self, parser):
        """The tree should contain a form node."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        forms = tree.get_nodes_by_role("form")
        assert len(forms) >= 1

    def test_contains_textboxes(self, parser):
        """The tree should contain textbox nodes for the inputs."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        textboxes = tree.get_nodes_by_role("textbox")
        assert len(textboxes) >= 2

    def test_contains_button(self, parser):
        """The tree should contain a button node."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        buttons = tree.get_nodes_by_role("button")
        assert len(buttons) >= 1

    def test_button_has_name(self, parser):
        """The button should have 'Submit' as its name."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        buttons = tree.get_nodes_by_role("button")
        assert any(b.name == "Submit" for b in buttons)

    def test_tree_validates(self, parser):
        """The parsed tree should pass structural validation."""
        json_str = _load_json("simple_form.json")
        tree = parser.parse(json_str)
        assert tree.validate() == []


# ── parse_dict() ──────────────────────────────────────────────────────────────


class TestParseDict:
    """Tests for the parse_dict() method that accepts Python dicts."""

    def test_parse_dict_returns_tree(self, parser):
        """parse_dict() with a dict should return an AccessibilityTree."""
        data = _load_json_dict("simple_form.json")
        tree = parser.parse_dict(data)
        assert isinstance(tree, AccessibilityTree)

    def test_parse_dict_matches_parse(self, parser):
        """parse_dict(data) should produce same tree size as parse(json_str)."""
        json_str = _load_json("simple_form.json")
        data = json.loads(json_str)
        tree_str = parser.parse(json_str)
        tree_dict = parser.parse_dict(data)
        assert tree_str.size() == tree_dict.size()

    def test_parse_dict_preserves_names(self, parser):
        """parse_dict should preserve node names from the dict."""
        data = _load_json_dict("simple_form.json")
        tree = parser.parse_dict(data)
        assert tree.root.name == "Login Page"


# ── Navigation menu JSON ─────────────────────────────────────────────────────


class TestNavigationJSON:
    """Tests for parsing the navigation_menu.json fixture."""

    def test_contains_navigation(self, parser):
        """The navigation JSON should produce a navigation node."""
        json_str = _load_json("navigation_menu.json")
        tree = parser.parse(json_str)
        navs = tree.get_nodes_by_role("navigation")
        assert len(navs) >= 1

    def test_contains_links(self, parser):
        """The navigation should contain link nodes."""
        json_str = _load_json("navigation_menu.json")
        tree = parser.parse(json_str)
        links = tree.get_nodes_by_role("link")
        assert len(links) >= 4

    def test_link_names(self, parser):
        """Links should have recognisable names like Home, Products, etc."""
        json_str = _load_json("navigation_menu.json")
        tree = parser.parse(json_str)
        links = tree.get_nodes_by_role("link")
        names = {l.name for l in links}
        assert "Home" in names


# ── Dashboard JSON ────────────────────────────────────────────────────────────


class TestDashboardJSON:
    """Tests for parsing the complex_dashboard.json fixture."""

    def test_dashboard_parses(self, parser):
        """The dashboard JSON should parse without error."""
        json_str = _load_json("complex_dashboard.json")
        tree = parser.parse(json_str)
        assert tree.size() > 5

    def test_dashboard_has_heading(self, parser):
        """The dashboard should contain at least one heading."""
        json_str = _load_json("complex_dashboard.json")
        tree = parser.parse(json_str)
        headings = tree.get_nodes_by_role("heading")
        assert len(headings) >= 1

    def test_dashboard_validates(self, parser):
        """The dashboard tree should pass structural validation."""
        json_str = _load_json("complex_dashboard.json")
        tree = parser.parse(json_str)
        assert tree.validate() == []


# ── Modal dialog JSON ─────────────────────────────────────────────────────────


class TestModalDialogJSON:
    """Tests for parsing the modal_dialog.json fixture."""

    def test_modal_parses(self, parser):
        """The modal dialog JSON should parse without error."""
        json_str = _load_json("modal_dialog.json")
        tree = parser.parse(json_str)
        assert tree.root is not None

    def test_modal_has_dialog(self, parser):
        """The modal tree should contain a dialog role."""
        json_str = _load_json("modal_dialog.json")
        tree = parser.parse(json_str)
        dialogs = tree.get_nodes_by_role("dialog")
        assert len(dialogs) >= 1


# ── Format detection ──────────────────────────────────────────────────────────


class TestFormatDetection:
    """Tests for _detect_format() heuristic."""

    def test_generic_format_for_role_children(self):
        """A dict with 'role' and 'children' should be detected as generic."""
        data = {"role": "document", "children": []}
        assert _detect_format(data) == _Format.GENERIC

    def test_chrome_devtools_format(self):
        """A dict with 'nodes' list containing 'nodeId' should be chrome_devtools."""
        data = {"nodes": [{"nodeId": 1, "role": {"value": "document"}}]}
        assert _detect_format(data) == _Format.CHROME_DEVTOOLS

    def test_generic_format_for_list_of_roles(self):
        """A list of dicts with 'role' should be detected as generic."""
        data = [{"role": "button", "name": "OK"}]
        assert _detect_format(data) == _Format.GENERIC

    def test_chrome_devtools_list_format(self):
        """A list of dicts with 'nodeId' should be chrome_devtools."""
        data = [{"nodeId": 1, "role": {"value": "document"}}]
        assert _detect_format(data) == _Format.CHROME_DEVTOOLS

    def test_axe_core_format(self):
        """A dict with 'violations' key should be axe_core."""
        data = {"violations": [], "passes": []}
        assert _detect_format(data) == _Format.AXE_CORE

    def test_empty_dict_is_generic(self):
        """An empty dict should fall back to generic."""
        assert _detect_format({}) == _Format.GENERIC


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    """Tests for invalid/malformed input handling."""

    def test_invalid_json_string_raises(self, parser):
        """parse() with invalid JSON should raise an appropriate error."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parser.parse("not valid json {{{")

    def test_empty_json_string(self, parser):
        """parse() with an empty object should handle gracefully."""
        # An empty dict should still go through the generic parser path
        try:
            tree = parser.parse("{}")
            # If it doesn't raise, the tree should exist
            assert tree is not None
        except (ValueError, KeyError, TypeError):
            pass  # Acceptable to raise on empty input

    def test_parse_dict_with_minimal_data(self, parser):
        """parse_dict with minimal valid data should produce a tree."""
        data = {"role": "document", "name": "Minimal", "children": []}
        tree = parser.parse_dict(data)
        assert tree.root is not None
        assert tree.root.role == "document"

    def test_parse_dict_missing_role_handles_gracefully(self, parser):
        """parse_dict with missing role should handle gracefully."""
        data = {"name": "NoRole", "children": []}
        try:
            tree = parser.parse_dict(data)
            # If it succeeds, it should still have a root
            assert tree.root is not None
        except (KeyError, ValueError, TypeError):
            pass  # Acceptable to raise on missing role

    def test_parse_file(self, parser, tmp_path):
        """parse_file should read from a file path."""
        p = tmp_path / "test.json"
        data = {"role": "document", "name": "File Test", "children": []}
        p.write_text(json.dumps(data))
        tree = parser.parse_file(p)
        assert tree.root is not None
        assert tree.root.name == "File Test"


# ── ID prefix ────────────────────────────────────────────────────────────────


class TestIDPrefix:
    """Tests for custom ID prefix configuration."""

    def test_default_prefix(self):
        """Default parser should use 'j' prefix for generated IDs."""
        parser = JSONAccessibilityParser()
        data = {"role": "document", "name": "Test", "children": []}
        tree = parser.parse_dict(data)
        # IDs generated by the parser should start with 'j'
        assert tree.root.id.startswith("j")

    def test_custom_prefix(self):
        """Custom id_prefix should be used for generated IDs."""
        parser = JSONAccessibilityParser(id_prefix="custom")
        data = {"role": "document", "name": "Test", "children": []}
        tree = parser.parse_dict(data)
        assert tree.root.id.startswith("custom")
