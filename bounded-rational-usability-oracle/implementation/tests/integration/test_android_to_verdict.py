"""Integration test: Android XML → accessibility tree → cost → verdict.

Exercises the end-to-end pipeline from Android view hierarchy XML through
to regression verdicts. Tests with login screen mutations: added buttons,
moved buttons, and equivalent restructuring.
"""

from __future__ import annotations

import pytest
import numpy as np

from usability_oracle.android_a11y.parser import AndroidAccessibilityParser
from usability_oracle.android_a11y.converter import AndroidToAccessibilityConverter
from usability_oracle.android_a11y.conformance import run_all_checks
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.accessibility.models import AccessibilityTree
from usability_oracle.mdp.builder import MDPBuilder
from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.comparison.paired import PairedComparator
from usability_oracle.comparison.models import ComparisonResult
from usability_oracle.comparison.models import AlignmentResult as CompAlignmentResult
from usability_oracle.comparison.models import StateMapping
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Android XML Fixtures (string literals)
# ---------------------------------------------------------------------------

LOGIN_SCREEN_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        package="com.example" content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="false" focused="false"
        scrollable="false" long-clickable="false" password="false" selected="false"
        bounds="[0,0][1080,1920]">
    <node index="0" text="Login" resource-id="com.example:id/title"
          class="android.widget.TextView" package="com.example"
          content-desc="Login screen title" checkable="false" checked="false"
          clickable="false" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[100,100][980,200]" />
    <node index="1" text="" resource-id="com.example:id/username"
          class="android.widget.EditText" package="com.example"
          content-desc="Username input" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[100,250][980,350]" />
    <node index="2" text="" resource-id="com.example:id/password"
          class="android.widget.EditText" package="com.example"
          content-desc="Password input" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="true" password="true" selected="false"
          bounds="[100,400][980,500]" />
    <node index="3" text="Sign In" resource-id="com.example:id/login_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Sign in button" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[300,550][780,650]" />
  </node>
</hierarchy>
"""

# Added extra buttons → Hick's regression
MANY_BUTTONS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        package="com.example" content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="false" focused="false"
        scrollable="false" long-clickable="false" password="false" selected="false"
        bounds="[0,0][1080,1920]">
    <node index="0" text="Login" resource-id="com.example:id/title"
          class="android.widget.TextView" package="com.example"
          content-desc="Login screen title" checkable="false" checked="false"
          clickable="false" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[100,100][980,200]" />
    <node index="1" text="" resource-id="com.example:id/username"
          class="android.widget.EditText" package="com.example"
          content-desc="Username input" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[100,250][980,350]" />
    <node index="2" text="" resource-id="com.example:id/password"
          class="android.widget.EditText" package="com.example"
          content-desc="Password input" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="true" password="true" selected="false"
          bounds="[100,400][980,500]" />
    <node index="3" text="Sign In" resource-id="com.example:id/login_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Sign in button" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[300,550][780,650]" />
    <node index="4" text="Sign Up" resource-id="com.example:id/signup_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Sign up button" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[300,700][780,800]" />
    <node index="5" text="Forgot Password" resource-id="com.example:id/forgot_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Forgot password" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[300,850][780,950]" />
    <node index="6" text="Guest" resource-id="com.example:id/guest_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Guest login" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[300,1000][780,1100]" />
    <node index="7" text="SSO" resource-id="com.example:id/sso_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Single sign-on" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[300,1150][780,1250]" />
  </node>
</hierarchy>
"""

# Moved button — smaller and farther → Fitts' regression
MOVED_BUTTON_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        package="com.example" content-desc="" checkable="false" checked="false"
        clickable="false" enabled="true" focusable="false" focused="false"
        scrollable="false" long-clickable="false" password="false" selected="false"
        bounds="[0,0][1080,1920]">
    <node index="0" text="Login" resource-id="com.example:id/title"
          class="android.widget.TextView" package="com.example"
          content-desc="Login screen title" checkable="false" checked="false"
          clickable="false" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[100,100][980,200]" />
    <node index="1" text="" resource-id="com.example:id/username"
          class="android.widget.EditText" package="com.example"
          content-desc="Username input" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[100,250][980,350]" />
    <node index="2" text="" resource-id="com.example:id/password"
          class="android.widget.EditText" package="com.example"
          content-desc="Password input" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="true" password="true" selected="false"
          bounds="[100,400][980,500]" />
    <node index="3" text="Sign In" resource-id="com.example:id/login_btn"
          class="android.widget.Button" package="com.example"
          content-desc="Sign in button" checkable="false" checked="false"
          clickable="true" enabled="true" focusable="true" focused="false"
          scrollable="false" long-clickable="false" password="false" selected="false"
          bounds="[950,1800][1050,1850]" />
  </node>
</hierarchy>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_android(xml: str) -> AccessibilityTree:
    """Parse Android XML → ViewHierarchy → AccessibilityTree."""
    parser = AndroidAccessibilityParser()
    hierarchy = parser.parse_xml(xml)
    converter = AndroidToAccessibilityConverter()
    tree = converter.convert(hierarchy)
    normaliser = AccessibilityNormalizer()
    tree = normaliser.normalize(tree)
    tree.root.properties['tabindex'] = '0'
    return tree


def _make_login_task() -> TaskSpec:
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus username"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Sign In", description="Submit",
                 depends_on=["s2"]),
    ]
    flow = TaskFlow(flow_id="login", name="Login", steps=steps,
                    success_criteria=["logged_in"])
    return TaskSpec(spec_id="login", name="Login", flows=[flow])


def _make_comp_alignment(mdp_a: MDP, mdp_b: MDP) -> CompAlignmentResult:
    common = set(mdp_a.states.keys()) & set(mdp_b.states.keys())
    mappings = [StateMapping(state_a=s, state_b=s) for s in common]
    if not mappings:
        mappings = [StateMapping(
            state_a=list(mdp_a.states.keys())[0],
            state_b=list(mdp_b.states.keys())[0],
        )]
    return CompAlignmentResult(mappings=mappings)


def _compare_android(xml_a: str, xml_b: str, task: TaskSpec) -> ComparisonResult:
    """Full pipeline: Android XML pair → comparison result."""
    tree_a = _parse_android(xml_a)
    tree_b = _parse_android(xml_b)
    builder = MDPBuilder()
    mdp_a = builder.build(tree_a, task)
    mdp_b = builder.build(tree_b, task)
    alignment = _make_comp_alignment(mdp_a, mdp_b)
    return PairedComparator().compare(
        mdp_a=mdp_a, mdp_b=mdp_b, alignment=alignment, task=task,
    )


# ===================================================================
# Tests — Parse stage
# ===================================================================


class TestAndroidXMLParsing:
    """Verify Android XML parses into valid accessibility trees."""

    def test_login_screen_parses(self):
        tree = _parse_android(LOGIN_SCREEN_XML)
        assert tree.size() > 0

    def test_many_buttons_parses(self):
        tree = _parse_android(MANY_BUTTONS_XML)
        assert tree.size() > 0

    def test_moved_button_parses(self):
        tree = _parse_android(MOVED_BUTTON_XML)
        assert tree.root is not None

    def test_many_buttons_has_more_nodes(self):
        simple = _parse_android(LOGIN_SCREEN_XML)
        many = _parse_android(MANY_BUTTONS_XML)
        assert many.size() > simple.size()

    def test_raw_hierarchy_node_count(self):
        """ViewHierarchy should count all XML nodes."""
        parser = AndroidAccessibilityParser()
        hierarchy = parser.parse_xml(LOGIN_SCREEN_XML)
        assert hierarchy.node_count >= 4


# ===================================================================
# Tests — Hick's regression (too many choices)
# ===================================================================


class TestAndroidHicksRegression:
    """Adding buttons → Hick's regression if too many choices."""

    def test_many_buttons_not_improvement(self):
        task = _make_login_task()
        result = _compare_android(LOGIN_SCREEN_XML, MANY_BUTTONS_XML, task)
        assert result.verdict != RegressionVerdict.IMPROVEMENT

    def test_many_buttons_mdp_bigger(self):
        task = _make_login_task()
        mdp_simple = MDPBuilder().build(_parse_android(LOGIN_SCREEN_XML), task)
        mdp_many = MDPBuilder().build(_parse_android(MANY_BUTTONS_XML), task)
        assert mdp_many.n_states >= mdp_simple.n_states


# ===================================================================
# Tests — Fitts' regression (button moved/resized)
# ===================================================================


class TestAndroidFittsRegression:
    """Moving button farther/smaller → Fitts' regression."""

    def test_moved_button_not_improvement(self):
        task = _make_login_task()
        result = _compare_android(LOGIN_SCREEN_XML, MOVED_BUTTON_XML, task)
        # Button moved to corner, smaller — should not be improvement
        assert result.verdict != RegressionVerdict.IMPROVEMENT


# ===================================================================
# Tests — Equivalent restructuring → PASS
# ===================================================================


class TestAndroidEquivalentRestructuring:
    """Identical XML should produce PASS / NEUTRAL verdict."""

    def test_same_xml_neutral(self):
        task = _make_login_task()
        result = _compare_android(LOGIN_SCREEN_XML, LOGIN_SCREEN_XML, task)
        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )

    def test_same_xml_costs_equal(self):
        task = _make_login_task()
        result = _compare_android(LOGIN_SCREEN_XML, LOGIN_SCREEN_XML, task)
        if result.cost_before is not None and result.cost_after is not None:
            cb = result.cost_before.mean_time
            ca = result.cost_after.mean_time
            assert abs(cb - ca) < 1.0

    def test_same_xml_small_effect_size(self):
        task = _make_login_task()
        result = _compare_android(LOGIN_SCREEN_XML, LOGIN_SCREEN_XML, task)
        assert abs(result.effect_size) < 1.0


# ===================================================================
# Tests — Conformance check integration
# ===================================================================


class TestAndroidConformance:
    """Conformance checks on parsed Android hierarchies."""

    def test_login_screen_conformance(self):
        parser = AndroidAccessibilityParser()
        hierarchy = parser.parse_xml(LOGIN_SCREEN_XML)
        results = run_all_checks(hierarchy)
        assert isinstance(results, list)

    def test_many_buttons_conformance(self):
        parser = AndroidAccessibilityParser()
        hierarchy = parser.parse_xml(MANY_BUTTONS_XML)
        results = run_all_checks(hierarchy)
        assert isinstance(results, list)
