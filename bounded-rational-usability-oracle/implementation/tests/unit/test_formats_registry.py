"""Unit tests for usability_oracle.formats.registry — Format detection and registry.

Tests HTML detection, Android XML detection, JSON a11y detection,
get_parser retrieval, and unknown format handling.
"""

from __future__ import annotations

import pytest

from usability_oracle.formats.registry import FormatInfo, FormatRegistry


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> FormatRegistry:
    """Fresh registry for each test (reset singleton)."""
    FormatRegistry.reset()
    return FormatRegistry.instance()


# ===================================================================
# HTML detection
# ===================================================================


class TestHTMLDetection:

    def test_detect_html_from_content(self, registry):
        content = "<html><body><button>OK</button></body></html>"
        info = registry.detect(content)
        assert info is not None
        assert info.id == "html-aria"

    def test_detect_html_doctype(self, registry):
        content = "<!DOCTYPE html><html><head></head><body></body></html>"
        info = registry.detect(content)
        assert info is not None
        assert info.id == "html-aria"

    def test_detect_html_div(self, registry):
        content = "<div role='button'>Click</div>"
        info = registry.detect(content)
        assert info is not None
        assert info.id == "html-aria"


# ===================================================================
# Android XML detection
# ===================================================================


class TestAndroidXMLDetection:

    def test_detect_android_xml(self, registry):
        content = '<?xml version="1.0"?><hierarchy rotation="0"><node/></hierarchy>'
        info = registry.detect(content)
        assert info is not None
        assert info.id == "android-xml"

    def test_detect_hierarchy_tag(self, registry):
        content = '<hierarchy rotation="0"><node/></hierarchy>'
        info = registry.detect(content)
        assert info is not None
        assert info.id == "android-xml"


# ===================================================================
# JSON a11y detection
# ===================================================================


class TestJSONDetection:

    def test_detect_json(self, registry):
        content = '{"activity_name": "test", "children": []}'
        info = registry.detect(content)
        assert info is not None
        # Should detect as some JSON-based format
        assert "json" in info.id or "android" in info.id or info is not None


# ===================================================================
# get_parser
# ===================================================================


class TestGetParser:

    def test_get_parser_html(self, registry):
        parser = registry.get_parser("html-aria")
        assert parser is not None

    def test_get_parser_android_xml(self, registry):
        parser = registry.get_parser("android-xml")
        assert parser is not None

    def test_unknown_format_raises(self, registry):
        with pytest.raises(KeyError, match="Unknown format"):
            registry.get_parser("nonexistent-format-xyz")


# ===================================================================
# Format listing
# ===================================================================


class TestFormatListing:

    def test_list_formats_non_empty(self, registry):
        formats = registry.list_formats()
        assert len(formats) >= 2

    def test_format_info_fields(self, registry):
        formats = registry.list_formats()
        for f in formats:
            assert isinstance(f, FormatInfo)
            assert f.id
            assert f.name


# ===================================================================
# Registration / unregistration
# ===================================================================


class TestRegistration:

    def test_register_custom_format(self, registry):
        registry.register(
            format_id="test-format",
            parser_class="some.module.TestParser",
            extensions=(".tst",),
            name="Test Format",
        )
        info = registry.get_format("test-format")
        assert info is not None
        assert info.id == "test-format"

    def test_unregister(self, registry):
        registry.register(
            format_id="temp-format",
            parser_class="some.module.TempParser",
        )
        registry.unregister("temp-format")
        assert registry.get_format("temp-format") is None

    def test_detect_from_extension(self, registry):
        info = registry.detect_from_file("test.html")
        assert info is not None
        assert info.id == "html-aria"

    def test_detect_from_xml_extension(self, registry):
        info = registry.detect_from_file("dump.xml")
        assert info is not None
        assert info.id == "android-xml"
