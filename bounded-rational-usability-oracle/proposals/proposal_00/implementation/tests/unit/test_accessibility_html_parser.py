"""Tests for usability_oracle.accessibility.html_parser — HTML to AccessibilityTree.

Covers the HTMLAccessibilityParser.parse() method with various HTML structures,
role inference from semantic HTML elements, name extraction from labels and
ARIA attributes, and edge cases such as empty/malformed HTML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.models import AccessibilityTree

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_html"


def _load_html(name: str) -> str:
    """Load a sample HTML fixture by filename."""
    return (_FIXTURES / name).read_text()


@pytest.fixture
def parser() -> HTMLAccessibilityParser:
    """Return a default HTMLAccessibilityParser instance."""
    return HTMLAccessibilityParser()


# ── Simple form parsing ──────────────────────────────────────────────────────


class TestSimpleFormParsing:
    """Tests for parsing the simple_form.html fixture."""

    def test_parse_returns_tree(self, parser):
        """parse() should return an AccessibilityTree."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        assert isinstance(tree, AccessibilityTree)

    def test_tree_has_root(self, parser):
        """The parsed tree should have a root node."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        assert tree.root is not None

    def test_tree_contains_form(self, parser):
        """The parsed tree should contain a form role node."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        forms = tree.get_nodes_by_role("form")
        assert len(forms) >= 1

    def test_tree_contains_textboxes(self, parser):
        """The form should contain textbox nodes for username/password inputs."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        textboxes = tree.get_nodes_by_role("textbox")
        assert len(textboxes) >= 2

    def test_tree_contains_button(self, parser):
        """The form should contain a submit button."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        buttons = tree.get_nodes_by_role("button")
        assert len(buttons) >= 1

    def test_tree_contains_checkbox(self, parser):
        """The form should contain a checkbox for 'Remember me'."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        checkboxes = tree.get_nodes_by_role("checkbox")
        assert len(checkboxes) >= 1

    def test_form_has_aria_label(self, parser):
        """The form node should have 'Login' as its name from aria-label."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        forms = tree.get_nodes_by_role("form")
        names = [f.name for f in forms]
        assert any("Login" in n for n in names)

    def test_tree_is_valid(self, parser):
        """The parsed tree should pass structural validation."""
        html = _load_html("simple_form.html")
        tree = parser.parse(html)
        errors = tree.validate()
        assert errors == []


# ── Navigation menu parsing ──────────────────────────────────────────────────


class TestNavigationParsing:
    """Tests for parsing the navigation_menu.html fixture."""

    def test_contains_navigation_role(self, parser):
        """The tree should contain a navigation landmark."""
        html = _load_html("navigation_menu.html")
        tree = parser.parse(html)
        navs = tree.get_nodes_by_role("navigation")
        assert len(navs) >= 1

    def test_contains_menubar(self, parser):
        """The navigation should contain a menubar."""
        html = _load_html("navigation_menu.html")
        tree = parser.parse(html)
        menubars = tree.get_nodes_by_role("menubar")
        assert len(menubars) >= 1

    def test_contains_menuitems(self, parser):
        """The menubar should contain multiple menuitem nodes."""
        html = _load_html("navigation_menu.html")
        tree = parser.parse(html)
        items = tree.get_nodes_by_role("menuitem")
        assert len(items) >= 5  # Home, Products, Services, About, Contact

    def test_submenu_exists(self, parser):
        """The Services item should have a submenu."""
        html = _load_html("navigation_menu.html")
        tree = parser.parse(html)
        menus = tree.get_nodes_by_role("menu")
        assert len(menus) >= 1

    def test_navigation_label(self, parser):
        """The navigation should have 'Main Navigation' as its name."""
        html = _load_html("navigation_menu.html")
        tree = parser.parse(html)
        navs = tree.get_nodes_by_role("navigation")
        assert any("Main Navigation" in n.name for n in navs)


# ── Modal dialog parsing ─────────────────────────────────────────────────────


class TestModalDialogParsing:
    """Tests for parsing the modal_dialog.html fixture."""

    def test_contains_dialog(self, parser):
        """The tree should contain a dialog role."""
        html = _load_html("modal_dialog.html")
        tree = parser.parse(html)
        dialogs = tree.get_nodes_by_role("dialog")
        assert len(dialogs) >= 1

    def test_dialog_has_buttons(self, parser):
        """The dialog should contain Delete and Cancel buttons."""
        html = _load_html("modal_dialog.html")
        tree = parser.parse(html)
        buttons = tree.get_nodes_by_role("button")
        assert len(buttons) >= 2

    def test_dialog_has_heading(self, parser):
        """The dialog should contain a heading node."""
        html = _load_html("modal_dialog.html")
        tree = parser.parse(html)
        headings = tree.get_nodes_by_role("heading")
        assert len(headings) >= 1


# ── Dashboard parsing ────────────────────────────────────────────────────────


class TestDashboardParsing:
    """Tests for parsing the complex_dashboard.html fixture."""

    def test_dashboard_parses(self, parser):
        """The dashboard HTML should parse without error."""
        html = _load_html("complex_dashboard.html")
        tree = parser.parse(html)
        assert tree.size() > 0

    def test_dashboard_valid(self, parser):
        """The dashboard tree should pass validation."""
        html = _load_html("complex_dashboard.html")
        tree = parser.parse(html)
        assert tree.validate() == []


# ── Role inference ────────────────────────────────────────────────────────────


class TestRoleInference:
    """Tests for implicit role mapping from HTML elements."""

    def test_button_element_maps_to_button_role(self, parser):
        """A <button> element should map to the 'button' role."""
        tree = parser.parse("<html><body><button>Click</button></body></html>")
        buttons = tree.get_nodes_by_role("button")
        assert len(buttons) >= 1

    def test_input_text_maps_to_textbox(self, parser):
        """An <input type='text'> should map to the 'textbox' role."""
        tree = parser.parse('<html><body><input type="text"></body></html>')
        textboxes = tree.get_nodes_by_role("textbox")
        assert len(textboxes) >= 1

    def test_select_maps_to_combobox(self, parser):
        """A <select> element should map to 'combobox' role."""
        html = "<html><body><select><option>A</option></select></body></html>"
        tree = parser.parse(html)
        combos = tree.get_nodes_by_role("combobox")
        assert len(combos) >= 1

    def test_anchor_maps_to_link(self, parser):
        """An <a href> element should map to 'link' role."""
        tree = parser.parse('<html><body><a href="/">Home</a></body></html>')
        links = tree.get_nodes_by_role("link")
        assert len(links) >= 1

    def test_nav_maps_to_navigation(self, parser):
        """A <nav> element should map to 'navigation' role."""
        tree = parser.parse("<html><body><nav>Menu</nav></body></html>")
        navs = tree.get_nodes_by_role("navigation")
        assert len(navs) >= 1

    def test_input_checkbox_maps_to_checkbox(self, parser):
        """An <input type='checkbox'> should map to 'checkbox' role."""
        tree = parser.parse('<html><body><input type="checkbox"></body></html>')
        checks = tree.get_nodes_by_role("checkbox")
        assert len(checks) >= 1

    def test_input_radio_maps_to_radio(self, parser):
        """An <input type='radio'> should map to 'radio' role."""
        tree = parser.parse('<html><body><input type="radio"></body></html>')
        radios = tree.get_nodes_by_role("radio")
        assert len(radios) >= 1

    def test_textarea_maps_to_textbox(self, parser):
        """A <textarea> element should map to 'textbox' role."""
        tree = parser.parse("<html><body><textarea></textarea></body></html>")
        textboxes = tree.get_nodes_by_role("textbox")
        assert len(textboxes) >= 1

    def test_explicit_role_overrides_implicit(self, parser):
        """An explicit role= attribute should override the implicit mapping."""
        tree = parser.parse('<html><body><div role="button">Act</div></body></html>')
        buttons = tree.get_nodes_by_role("button")
        assert len(buttons) >= 1


# ── Name extraction ───────────────────────────────────────────────────────────


class TestNameExtraction:
    """Tests for accessible name derivation from labels, aria-label, title, alt."""

    def test_aria_label(self, parser):
        """aria-label should be used as the accessible name."""
        tree = parser.parse('<html><body><button aria-label="Close">X</button></body></html>')
        buttons = tree.get_nodes_by_role("button")
        assert any("Close" in b.name for b in buttons)

    def test_title_attribute(self, parser):
        """title attribute should contribute to name if no aria-label."""
        tree = parser.parse('<html><body><button title="Save file">💾</button></body></html>')
        buttons = tree.get_nodes_by_role("button")
        assert any(b.name for b in buttons)

    def test_alt_on_img(self, parser):
        """alt text on <img> should become the accessible name."""
        tree = parser.parse('<html><body><img alt="Company Logo"></body></html>')
        imgs = tree.get_nodes_by_role("img")
        assert any("Company Logo" in i.name for i in imgs)

    def test_text_content_as_name(self, parser):
        """Text content of a button should become its name."""
        tree = parser.parse("<html><body><button>Submit Form</button></body></html>")
        buttons = tree.get_nodes_by_role("button")
        assert any("Submit" in b.name for b in buttons)


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases: empty HTML, malformed markup, etc."""

    def test_empty_html(self, parser):
        """Parsing empty HTML should produce a tree with at least a root."""
        tree = parser.parse("<html><body></body></html>")
        assert tree.root is not None
        assert tree.size() >= 1

    def test_minimal_html(self, parser):
        """Parsing minimal valid HTML should work."""
        tree = parser.parse("<html><body><p>Hello</p></body></html>")
        assert tree.size() >= 1

    def test_nested_structure_preserved(self, parser):
        """Deeply nested HTML should produce a tree with correct depth."""
        html = "<html><body><div><div><div><span>Deep</span></div></div></div></body></html>"
        tree = parser.parse(html)
        assert tree.depth() >= 3

    def test_no_text_nodes_option(self):
        """Parser with include_text_nodes=False should skip pure text nodes."""
        p = HTMLAccessibilityParser(include_text_nodes=False)
        tree = p.parse("<html><body><p>Hello world</p></body></html>")
        assert tree.root is not None
