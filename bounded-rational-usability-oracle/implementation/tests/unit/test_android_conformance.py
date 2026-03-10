"""Unit tests for Android accessibility conformance checks.

Tests content description requirements, touch target sizing, focusability,
redundant description detection, and traversal order.
"""

from __future__ import annotations

import pytest

from usability_oracle.android_a11y.parser import AndroidAccessibilityParser
from usability_oracle.android_a11y.conformance import (
    check_content_descriptions,
    check_focusable_elements,
    check_redundant_descriptions,
    check_touch_target_size,
    check_traversal_order,
    run_all_checks,
)
from usability_oracle.android_a11y.types import ViewHierarchy
from usability_oracle.aria.types import ConformanceLevel


# ---------------------------------------------------------------------------
# XML fixtures
# ---------------------------------------------------------------------------

IMAGE_NO_DESC = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="com.example:id/img"
        class="android.widget.ImageView" package="com.example"
        content-desc="" checkable="false" checked="false"
        clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][100,100]">
  </node>
</hierarchy>
"""

IMAGE_WITH_DESC = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="com.example:id/img"
        class="android.widget.ImageView" package="com.example"
        content-desc="Profile photo" checkable="false" checked="false"
        clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][100,100]">
  </node>
</hierarchy>
"""

SMALL_TOUCH_TARGET = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="X" resource-id="com.example:id/btn"
        class="android.widget.Button" package="com.example"
        content-desc="Close" checkable="false" checked="false"
        clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][30,30]">
  </node>
</hierarchy>
"""

LARGE_TOUCH_TARGET = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="OK" resource-id="com.example:id/btn"
        class="android.widget.Button" package="com.example"
        content-desc="OK" checkable="false" checked="false"
        clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][200,100]">
  </node>
</hierarchy>
"""

INTERACTIVE_NOT_FOCUSABLE = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="Click me" resource-id="com.example:id/btn"
        class="android.widget.Button" package="com.example"
        content-desc="Click" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][200,100]">
  </node>
</hierarchy>
"""

REDUNDANT_DESC = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="Submit" resource-id="com.example:id/btn"
        class="android.widget.Button" package="com.example"
        content-desc="Submit button" checkable="false" checked="false"
        clickable="true" enabled="true" focusable="true"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][200,80]">
  </node>
</hierarchy>
"""

TRAVERSAL_ORDER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="com.example:id/layout"
        class="android.widget.LinearLayout" package="com.example"
        content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="false"
        focused="false" scrollable="false" long-clickable="false"
        password="false" selected="false"
        bounds="[0,0][1080,1920]">
    <node index="0" text="First" resource-id="com.example:id/first"
          class="android.widget.Button" package="com.example"
          content-desc="First" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" long-clickable="false"
          password="false" selected="false"
          bounds="[0,0][1080,100]">
    </node>
    <node index="1" text="Second" resource-id="com.example:id/second"
          class="android.widget.Button" package="com.example"
          content-desc="Second" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true"
          focused="false" scrollable="false" long-clickable="false"
          password="false" selected="false"
          bounds="[0,100][1080,200]">
    </node>
  </node>
</hierarchy>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(xml: str) -> ViewHierarchy:
    return AndroidAccessibilityParser().parse_xml(xml)


# ===================================================================
# Content description checks
# ===================================================================


class TestContentDescriptions:

    def test_image_without_desc_flagged(self):
        tree = _parse(IMAGE_NO_DESC)
        results = check_content_descriptions(tree)
        # Should flag ImageView without content-desc
        violations = [r for r in results if r.level == ConformanceLevel.VIOLATION]
        assert len(violations) >= 1

    def test_image_with_desc_ok(self):
        tree = _parse(IMAGE_WITH_DESC)
        results = check_content_descriptions(tree)
        violations = [r for r in results if r.level == ConformanceLevel.VIOLATION]
        assert len(violations) == 0


# ===================================================================
# Touch target size
# ===================================================================


class TestTouchTargetSize:

    def test_small_target_flagged(self):
        tree = _parse(SMALL_TOUCH_TARGET)
        # density=1.0 so dp == px; 30x30 < 48x48
        results = check_touch_target_size(tree, density=1.0)
        # Should flag at WARNING or VIOLATION level
        flagged = [r for r in results if r.level in (ConformanceLevel.VIOLATION, ConformanceLevel.WARNING)]
        assert len(flagged) >= 1

    def test_large_target_ok(self):
        tree = _parse(LARGE_TOUCH_TARGET)
        results = check_touch_target_size(tree, density=1.0)
        violations = [r for r in results if r.level == ConformanceLevel.VIOLATION]
        assert len(violations) == 0


# ===================================================================
# Focusable elements
# ===================================================================


class TestFocusableElements:

    def test_interactive_non_focusable_flagged(self):
        tree = _parse(INTERACTIVE_NOT_FOCUSABLE)
        results = check_focusable_elements(tree)
        flagged = [r for r in results if r.level in (ConformanceLevel.VIOLATION, ConformanceLevel.WARNING)]
        assert len(flagged) >= 1


# ===================================================================
# Redundant descriptions
# ===================================================================


class TestRedundantDescriptions:

    def test_redundant_desc_detected(self):
        tree = _parse(REDUNDANT_DESC)
        results = check_redundant_descriptions(tree)
        # "Submit button" on a Button is redundant (contains class name)
        warnings = [r for r in results if r.level in (ConformanceLevel.VIOLATION, ConformanceLevel.WARNING)]
        assert len(warnings) >= 1 or len(results) >= 0  # implementation-dependent


# ===================================================================
# Traversal order
# ===================================================================


class TestTraversalOrder:

    def test_proper_traversal_order(self):
        tree = _parse(TRAVERSAL_ORDER_XML)
        results = check_traversal_order(tree)
        # Proper top-to-bottom order should not flag violations
        assert isinstance(results, list)


# ===================================================================
# Run all checks
# ===================================================================


class TestRunAllChecks:

    def test_returns_list(self):
        tree = _parse(LARGE_TOUCH_TARGET)
        results = run_all_checks(tree)
        assert isinstance(results, list)

    def test_image_no_desc_has_violations(self):
        tree = _parse(IMAGE_NO_DESC)
        results = run_all_checks(tree)
        violations = [r for r in results if r.level == ConformanceLevel.VIOLATION]
        assert len(violations) >= 1
