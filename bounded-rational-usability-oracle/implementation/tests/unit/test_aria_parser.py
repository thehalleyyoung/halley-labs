"""Unit tests for usability_oracle.aria.parser — ARIA HTML parser.

Tests HTML parsing, role extraction, accessible name computation,
and landmark detection.
"""

from __future__ import annotations

import pytest

from usability_oracle.aria.parser import AriaHTMLParser, AriaNodeInfo, AriaTree


# ---------------------------------------------------------------------------
# Test HTML snippets
# ---------------------------------------------------------------------------

SIMPLE_BUTTON = """
<html>
<body>
  <button id="btn1">Click me</button>
</body>
</html>
"""

NAV_LANDMARK = """
<html>
<body>
  <header>
    <h1>Site Title</h1>
  </header>
  <nav aria-label="Main navigation">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
    </ul>
  </nav>
  <main>
    <p>Content here</p>
  </main>
</body>
</html>
"""

EXPLICIT_ROLE = """
<html>
<body>
  <div role="button" tabindex="0">Custom button</div>
</body>
</html>
"""

ARIA_LABEL = """
<html>
<body>
  <button aria-label="Close dialog">X</button>
</body>
</html>
"""

ARIA_LABELLEDBY = """
<html>
<body>
  <h2 id="heading1">Settings</h2>
  <div role="region" aria-labelledby="heading1">
    <p>Region content</p>
  </div>
</body>
</html>
"""

NESTED_LANDMARKS = """
<html>
<body>
  <nav aria-label="Primary">
    <nav aria-label="Secondary">
      <a href="/nested">Nested</a>
    </nav>
  </nav>
  <main>
    <article>
      <p>Article content</p>
    </article>
  </main>
</body>
</html>
"""

FORM_INPUTS = """
<html>
<body>
  <form>
    <label for="username">Username</label>
    <input type="text" id="username" name="username">
    <label for="password">Password</label>
    <input type="password" id="password" name="password">
    <button type="submit">Login</button>
  </form>
</body>
</html>
"""


# ===================================================================
# Simple parsing
# ===================================================================


class TestSimpleParsing:

    def test_parse_button(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(SIMPLE_BUTTON)
        assert isinstance(tree, AriaTree)
        assert tree.root is not None

    def test_button_role_extracted(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(SIMPLE_BUTTON)
        # Find the button node
        button = _find_by_role(tree.root, "button")
        assert button is not None

    def test_tree_size_positive(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(SIMPLE_BUTTON)
        assert tree.size > 0


# ===================================================================
# Implicit role extraction
# ===================================================================


class TestImplicitRoles:

    def test_nav_becomes_navigation(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(NAV_LANDMARK)
        nav = _find_by_role(tree.root, "navigation")
        assert nav is not None

    def test_main_becomes_main(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(NAV_LANDMARK)
        main = _find_by_role(tree.root, "main")
        assert main is not None

    def test_header_becomes_banner(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(NAV_LANDMARK)
        banner = _find_by_role(tree.root, "banner")
        assert banner is not None


# ===================================================================
# Explicit role override
# ===================================================================


class TestExplicitRole:

    def test_explicit_role_overrides_implicit(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(EXPLICIT_ROLE)
        button = _find_by_role(tree.root, "button")
        assert button is not None


# ===================================================================
# aria-label extraction
# ===================================================================


class TestAriaLabel:

    def test_aria_label_extracted(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(ARIA_LABEL)
        button = _find_by_role(tree.root, "button")
        assert button is not None
        assert "Close dialog" in button.accessible_name


# ===================================================================
# Accessible name computation (aria-labelledby)
# ===================================================================


class TestAccessibleName:

    def test_aria_labelledby_chain(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(ARIA_LABELLEDBY)
        region = _find_by_role(tree.root, "region")
        assert region is not None
        assert "Settings" in region.accessible_name


# ===================================================================
# Landmark detection
# ===================================================================


class TestLandmarkDetection:

    def test_landmarks_found(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(NAV_LANDMARK)
        assert len(tree.landmarks) >= 2  # nav + main at least

    def test_landmark_roles(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(NAV_LANDMARK)
        landmark_roles = {lm.role for lm in tree.landmarks}
        assert "navigation" in landmark_roles or "main" in landmark_roles


# ===================================================================
# Nested landmarks
# ===================================================================


class TestNestedLandmarks:

    def test_nested_landmarks_parsed(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(NESTED_LANDMARKS)
        assert tree.root is not None
        nav_nodes = _find_all_by_role(tree.root, "navigation")
        assert len(nav_nodes) >= 2


# ===================================================================
# Form input labels
# ===================================================================


class TestFormInputLabels:

    def test_form_parsed(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(FORM_INPUTS)
        assert tree.root is not None

    def test_input_nodes_present(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(FORM_INPUTS)
        textbox = _find_by_role(tree.root, "textbox")
        assert textbox is not None

    def test_button_in_form(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(FORM_INPUTS)
        button = _find_by_role(tree.root, "button")
        assert button is not None


# ===================================================================
# AriaTree API
# ===================================================================


class TestAriaTreeAPI:

    def test_get_node_by_id(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(SIMPLE_BUTTON)
        # All nodes should be in the index
        for nid, node in tree.node_index.items():
            assert tree.get_node(nid) is node

    def test_to_dict(self):
        parser = AriaHTMLParser()
        tree = parser.parse_html(SIMPLE_BUTTON)
        d = tree.to_dict()
        assert isinstance(d, dict)
        assert "root" in d


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _find_by_role(node: AriaNodeInfo, role: str) -> AriaNodeInfo | None:
    if node.role == role:
        return node
    for child in node.children:
        found = _find_by_role(child, role)
        if found is not None:
            return found
    return None


def _find_all_by_role(node: AriaNodeInfo, role: str) -> list[AriaNodeInfo]:
    results = []
    if node.role == role:
        results.append(node)
    for child in node.children:
        results.extend(_find_all_by_role(child, role))
    return results
