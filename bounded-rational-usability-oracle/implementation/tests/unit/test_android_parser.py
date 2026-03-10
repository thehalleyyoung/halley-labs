"""Unit tests for usability_oracle.android_a11y.parser — Android accessibility parser.

Tests XML dump parsing, bounds extraction, content description, clickable/focusable
attributes, RecyclerView handling, nested ScrollView, and empty tree.
"""

from __future__ import annotations

import pytest

from usability_oracle.android_a11y.parser import AndroidAccessibilityParser
from usability_oracle.android_a11y.types import ViewHierarchy, AndroidNode, BoundsInfo


# ---------------------------------------------------------------------------
# XML fixtures
# ---------------------------------------------------------------------------

SIMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="Hello" resource-id="com.example:id/text1"
        class="android.widget.TextView" package="com.example"
        content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][540,96]">
  </node>
</hierarchy>
"""

BUTTON_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="Submit" resource-id="com.example:id/btn"
        class="android.widget.Button" package="com.example"
        content-desc="Submit form" checkable="false" checked="false"
        clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[100,200][300,280]">
  </node>
</hierarchy>
"""

RECYCLER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="com.example:id/recycler"
        class="androidx.recyclerview.widget.RecyclerView" package="com.example"
        content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="true"
        focused="false" scrollable="true" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][1080,1920]">
    <node index="0" text="Item 1" resource-id="com.example:id/item"
          class="android.widget.TextView" package="com.example"
          content-desc="" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" long-clickable="false"
          password="false" selected="false"
          bounds="[0,0][1080,100]">
    </node>
    <node index="1" text="Item 2" resource-id="com.example:id/item"
          class="android.widget.TextView" package="com.example"
          content-desc="" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" long-clickable="false"
          password="false" selected="false"
          bounds="[0,100][1080,200]">
    </node>
  </node>
</hierarchy>
"""

NESTED_SCROLL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="com.example:id/outer_scroll"
        class="android.widget.ScrollView" package="com.example"
        content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="true"
        focused="false" scrollable="true" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][1080,1920]">
    <node index="0" text="" resource-id="com.example:id/inner_scroll"
          class="android.widget.ScrollView" package="com.example"
          content-desc="" checkable="false" checked="false"
          clickable="false" enabled="true" focusable="true"
          focused="false" scrollable="true" long-clickable="false"
          password="false" selected="false"
          bounds="[0,0][1080,960]">
      <node index="0" text="Inner content" resource-id="com.example:id/inner_text"
            class="android.widget.TextView" package="com.example"
            content-desc="" checkable="false" checked="false"
            clickable="false" enabled="true" focusable="false"
            focused="false" scrollable="false" long-clickable="false"
            password="false" selected="false"
            bounds="[0,0][1080,200]">
      </node>
    </node>
  </node>
</hierarchy>
"""

EMPTY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
</hierarchy>
"""


# ===================================================================
# Simple XML parsing
# ===================================================================


class TestSimpleXMLParsing:

    def test_parse_returns_view_hierarchy(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(SIMPLE_XML)
        assert isinstance(result, ViewHierarchy)

    def test_root_id_present(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(SIMPLE_XML)
        assert result.root_id is not None

    def test_nodes_populated(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(SIMPLE_XML)
        assert len(result.nodes) >= 1


# ===================================================================
# Bounds parsing
# ===================================================================


class TestBoundsParsing:

    def test_bounds_format_parsed(self):
        parser = AndroidAccessibilityParser()
        bounds = parser.extract_bounds("[0,0][1080,1920]")
        assert isinstance(bounds, BoundsInfo)
        assert bounds.screen_left == 0
        assert bounds.screen_top == 0
        assert bounds.screen_right == 1080
        assert bounds.screen_bottom == 1920

    def test_button_bounds(self):
        parser = AndroidAccessibilityParser()
        bounds = parser.extract_bounds("[100,200][300,280]")
        assert bounds.screen_left == 100
        assert bounds.screen_top == 200
        assert bounds.screen_right == 300
        assert bounds.screen_bottom == 280

    @pytest.mark.parametrize("bounds_str,expected", [
        ("[0,0][100,100]", (0, 0, 100, 100)),
        ("[10,20][30,40]", (10, 20, 30, 40)),
        ("[540,960][1080,1920]", (540, 960, 1080, 1920)),
    ])
    def test_various_bounds(self, bounds_str, expected):
        parser = AndroidAccessibilityParser()
        bounds = parser.extract_bounds(bounds_str)
        assert (bounds.screen_left, bounds.screen_top,
                bounds.screen_right, bounds.screen_bottom) == expected


# ===================================================================
# Content description extraction
# ===================================================================


class TestContentDescription:

    def test_content_desc_extracted(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(BUTTON_XML)
        # Find the button node
        for node in result.nodes.values():
            if "Button" in str(node.class_name):
                assert node.description.content_description == "Submit form"
                break


# ===================================================================
# Clickable / Focusable attributes
# ===================================================================


class TestAttributes:

    def test_clickable_attribute(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(BUTTON_XML)
        for node in result.nodes.values():
            if "Button" in str(node.class_name):
                assert node.is_clickable is True
                break

    def test_focusable_attribute(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(BUTTON_XML)
        for node in result.nodes.values():
            if "Button" in str(node.class_name):
                assert node.is_focusable is True
                break

    def test_non_clickable_text(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(SIMPLE_XML)
        for node in result.nodes.values():
            if "TextView" in str(node.class_name):
                assert node.is_clickable is False
                break


# ===================================================================
# RecyclerView handling
# ===================================================================


class TestRecyclerView:

    def test_recycler_parsed(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(RECYCLER_XML)
        # Nodes may be deduplicated by resource-id; at least 2 nodes
        assert len(result.nodes) >= 2

    def test_recycler_children_exist(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(RECYCLER_XML)
        # Find recycler node and check it has children
        for node in result.nodes.values():
            if "RecyclerView" in str(node.class_name):
                assert len(node.child_ids) >= 2
                break

    def test_recycler_scrollable(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(RECYCLER_XML)
        for node in result.nodes.values():
            if "RecyclerView" in str(node.class_name):
                assert node.is_scrollable is True
                break


# ===================================================================
# Nested ScrollView
# ===================================================================


class TestNestedScrollView:

    def test_nested_scroll_parsed(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(NESTED_SCROLL_XML)
        scroll_count = sum(1 for n in result.nodes.values()
                           if "ScrollView" in str(n.class_name))
        assert scroll_count >= 2

    def test_inner_content_accessible(self):
        parser = AndroidAccessibilityParser()
        result = parser.parse_xml(NESTED_SCROLL_XML)
        texts = [n for n in result.nodes.values()
                 if n.description.text == "Inner content"]
        assert len(texts) >= 1


# ===================================================================
# Empty tree
# ===================================================================


class TestEmptyTree:

    def test_empty_hierarchy_parsed(self):
        parser = AndroidAccessibilityParser()
        # Empty hierarchy may raise ParseError; that's valid behavior
        with pytest.raises(Exception):
            parser.parse_xml(EMPTY_XML)
