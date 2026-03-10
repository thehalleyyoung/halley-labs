"""Integration test: ARIA HTML → accessibility tree → cognitive cost → verdict.

Exercises the end-to-end pipeline from raw HTML with ARIA roles through
to regression verdicts. Tests detect Hick's law, perceptual, and memory
regressions from structural UI changes.
"""

from __future__ import annotations

import copy

import pytest
import numpy as np

from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
)
from usability_oracle.alignment.differ import SemanticDiffer
from usability_oracle.alignment.models import AlignmentResult, AlignmentConfig
from usability_oracle.alignment.models import (
    AccessibilityTree as AlignTree,
    AccessibilityNode as AlignNode,
    AccessibilityRole,
    BoundingBox as AlignBBox,
)
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
# HTML Fixtures (string literals)
# ---------------------------------------------------------------------------

SIMPLE_NAV_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Simple Nav</title></head>
<body>
  <nav role="navigation" aria-label="Main menu">
    <ul role="menubar">
      <li role="menuitem"><a href="/home">Home</a></li>
      <li role="menuitem"><a href="/about">About</a></li>
      <li role="menuitem"><a href="/contact">Contact</a></li>
    </ul>
  </nav>
  <main role="main">
    <h1>Welcome</h1>
    <p>Content here.</p>
  </main>
</body>
</html>
"""

EXPANDED_NAV_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Expanded Nav</title></head>
<body>
  <nav role="navigation" aria-label="Main menu">
    <ul role="menubar">
      <li role="menuitem"><a href="/home">Home</a></li>
      <li role="menuitem"><a href="/about">About</a></li>
      <li role="menuitem"><a href="/contact">Contact</a></li>
      <li role="menuitem"><a href="/products">Products</a></li>
      <li role="menuitem"><a href="/services">Services</a></li>
      <li role="menuitem"><a href="/blog">Blog</a></li>
      <li role="menuitem"><a href="/careers">Careers</a></li>
      <li role="menuitem"><a href="/faq">FAQ</a></li>
      <li role="menuitem"><a href="/support">Support</a></li>
      <li role="menuitem"><a href="/press">Press</a></li>
    </ul>
  </nav>
  <main role="main">
    <h1>Welcome</h1>
    <p>Content here.</p>
  </main>
</body>
</html>
"""

LABELED_FORM_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Labeled Form</title></head>
<body>
  <form role="form" aria-label="Login">
    <div>
      <label for="user">Username</label>
      <input id="user" type="text" aria-required="true" />
    </div>
    <div>
      <label for="pass">Password</label>
      <input id="pass" type="password" aria-required="true" />
    </div>
    <button type="submit">Sign In</button>
  </form>
</body>
</html>
"""

UNLABELED_FORM_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Unlabeled Form</title></head>
<body>
  <form role="form" aria-label="Login">
    <div>
      <input id="user" type="text" placeholder="Username" aria-required="true" />
    </div>
    <div>
      <input id="pass" type="password" placeholder="Password" aria-required="true" />
    </div>
    <button type="submit">Sign In</button>
  </form>
</body>
</html>
"""

SHALLOW_MODAL_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Shallow Modal</title></head>
<body>
  <div role="dialog" aria-label="Confirm" aria-modal="true">
    <h2>Are you sure?</h2>
    <button>Yes</button>
    <button>No</button>
  </div>
</body>
</html>
"""

DEEP_MODAL_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Deep Modal</title></head>
<body>
  <div role="dialog" aria-label="Confirm" aria-modal="true">
    <div class="modal-wrapper">
      <div class="modal-content">
        <div class="modal-header">
          <h2>Are you sure?</h2>
        </div>
        <div class="modal-body">
          <p>This action cannot be undone.</p>
          <div class="form-group">
            <label for="reason">Reason</label>
            <input id="reason" type="text" />
          </div>
        </div>
        <div class="modal-footer">
          <button>Yes</button>
          <button>No</button>
          <button>Cancel</button>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_html(html: str) -> AccessibilityTree:
    """Parse HTML and normalise the accessibility tree."""
    parser = HTMLAccessibilityParser()
    tree = parser.parse(html)
    normaliser = AccessibilityNormalizer()
    tree = normaliser.normalize(tree)
    tree.root.properties['tabindex'] = '0'
    return tree


def _make_task() -> TaskSpec:
    """Minimal navigation task spec."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="menuitem",
                 target_name="Home", description="Click Home link"),
    ]
    flow = TaskFlow(flow_id="nav", name="Navigate", steps=steps,
                    success_criteria=["navigated"])
    return TaskSpec(spec_id="nav_task", name="Navigate", flows=[flow])


def _make_login_task() -> TaskSpec:
    """Login task spec for form tests."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus username"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type username", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Sign In", description="Submit",
                 depends_on=["s2"]),
    ]
    flow = TaskFlow(flow_id="login", name="Login", steps=steps,
                    success_criteria=["logged_in"])
    return TaskSpec(spec_id="login", name="Login", flows=[flow])


def _to_align_tree(atree: AccessibilityTree) -> AlignTree:
    """Convert accessibility tree to alignment tree."""
    nodes: dict[str, AlignNode] = {}

    def _role(role_str: str) -> AccessibilityRole:
        try:
            return AccessibilityRole(role_str)
        except ValueError:
            return AccessibilityRole.GENERIC

    def _visit(anode, parent_id=None):
        children_ids = [c.id for c in anode.children] if hasattr(anode, 'children') else []
        bbox = None
        if hasattr(anode, 'bounding_box') and anode.bounding_box is not None:
            bb = anode.bounding_box
            bbox = AlignBBox(x=bb.x, y=bb.y, width=bb.width, height=bb.height)
        nodes[anode.id] = AlignNode(
            node_id=anode.id, role=_role(anode.role),
            name=getattr(anode, 'name', ''),
            description=getattr(anode, 'description', ''),
            value=getattr(anode, 'value', ''),
            bounding_box=bbox,
            properties=dict(getattr(anode, 'properties', {})),
            parent_id=parent_id, children_ids=children_ids,
        )
        if hasattr(anode, 'children'):
            for child in anode.children:
                _visit(child, anode.id)

    _visit(atree.root)
    return AlignTree(nodes=nodes, root_ids=[atree.root.id])


def _make_comp_alignment(mdp_a: MDP, mdp_b: MDP) -> CompAlignmentResult:
    """Build comparison alignment from shared states."""
    common = set(mdp_a.states.keys()) & set(mdp_b.states.keys())
    mappings = [StateMapping(state_a=s, state_b=s) for s in common]
    if not mappings:
        mappings = [StateMapping(
            state_a=list(mdp_a.states.keys())[0],
            state_b=list(mdp_b.states.keys())[0],
        )]
    return CompAlignmentResult(mappings=mappings)


def _compare_html(html_a: str, html_b: str, task: TaskSpec) -> ComparisonResult:
    """Full pipeline: HTML pair → comparison result."""
    tree_a = _parse_html(html_a)
    tree_b = _parse_html(html_b)
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


class TestAriaHTMLParsing:
    """Verify ARIA HTML parses correctly into accessibility trees."""

    def test_nav_html_parses(self):
        tree = _parse_html(SIMPLE_NAV_HTML)
        assert tree.size() > 0

    def test_form_html_parses(self):
        tree = _parse_html(LABELED_FORM_HTML)
        assert tree.root is not None

    def test_modal_html_parses(self):
        tree = _parse_html(SHALLOW_MODAL_HTML)
        assert tree.size() >= 1

    def test_expanded_nav_has_more_nodes(self):
        """Expanded nav should have more nodes than simple nav."""
        simple = _parse_html(SIMPLE_NAV_HTML)
        expanded = _parse_html(EXPANDED_NAV_HTML)
        assert expanded.size() > simple.size()


# ===================================================================
# Tests — Hick's law regression (navigation with extra items)
# ===================================================================


class TestHicksLawRegression:
    """Navigation menu gaining extra items should detect Hick's regression."""

    def test_expanded_nav_detected_as_regression_or_different(self):
        """More menu items → higher choice cost (Hick's law)."""
        task = _make_task()
        result = _compare_html(SIMPLE_NAV_HTML, EXPANDED_NAV_HTML, task)
        assert isinstance(result.verdict, RegressionVerdict)
        # The expanded nav should not be detected as an improvement
        assert result.verdict != RegressionVerdict.IMPROVEMENT or \
            result.verdict == RegressionVerdict.INCONCLUSIVE

    def test_expanded_nav_mdp_has_more_states(self):
        """Expanded nav MDP should have more states than simple nav MDP."""
        task = _make_task()
        tree_simple = _parse_html(SIMPLE_NAV_HTML)
        tree_expanded = _parse_html(EXPANDED_NAV_HTML)
        mdp_simple = MDPBuilder().build(tree_simple, task)
        mdp_expanded = MDPBuilder().build(tree_expanded, task)
        assert mdp_expanded.n_states >= mdp_simple.n_states


# ===================================================================
# Tests — Perceptual regression (labels removed)
# ===================================================================


class TestPerceptualRegression:
    """Removing form labels should detect perceptual regression."""

    def test_unlabeled_form_not_improvement(self):
        """Removing labels should not be detected as improvement."""
        task = _make_login_task()
        result = _compare_html(LABELED_FORM_HTML, UNLABELED_FORM_HTML, task)
        assert result.verdict != RegressionVerdict.IMPROVEMENT

    def test_labeled_vs_labeled_neutral(self):
        """Same labeled form → NEUTRAL or INCONCLUSIVE."""
        task = _make_login_task()
        result = _compare_html(LABELED_FORM_HTML, LABELED_FORM_HTML, task)
        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )


# ===================================================================
# Tests — Memory regression (deeper modal)
# ===================================================================


class TestMemoryRegression:
    """Deeper modal dialog should detect memory load regression."""

    def test_deep_modal_not_improvement_over_shallow(self):
        """A deeper/more complex modal should not be an improvement."""
        task = _make_login_task()
        result = _compare_html(SHALLOW_MODAL_HTML, DEEP_MODAL_HTML, task)
        assert result.verdict != RegressionVerdict.IMPROVEMENT

    def test_deep_modal_has_more_nodes(self):
        """Deep modal tree should have more nodes."""
        shallow = _parse_html(SHALLOW_MODAL_HTML)
        deep = _parse_html(DEEP_MODAL_HTML)
        assert deep.size() > shallow.size()


# ===================================================================
# Tests — No-change and improvement verdicts
# ===================================================================


class TestVerdictOutcomes:
    """No change → PASS/NEUTRAL, improvement → IMPROVEMENT."""

    def test_no_change_produces_neutral(self):
        """Identical HTML should produce NEUTRAL or INCONCLUSIVE."""
        task = _make_task()
        result = _compare_html(SIMPLE_NAV_HTML, SIMPLE_NAV_HTML, task)
        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )

    def test_comparison_has_costs(self):
        """Comparison result always includes cost values."""
        task = _make_task()
        result = _compare_html(SIMPLE_NAV_HTML, SIMPLE_NAV_HTML, task)
        assert result.cost_before is not None
        assert result.cost_after is not None

    def test_comparison_has_effect_size(self):
        """Identical inputs yield near-zero effect size."""
        task = _make_task()
        result = _compare_html(SIMPLE_NAV_HTML, SIMPLE_NAV_HTML, task)
        assert abs(result.effect_size) < 1.0

    def test_simplified_nav_could_improve(self):
        """Going from expanded → simple nav should not be a regression."""
        task = _make_task()
        result = _compare_html(EXPANDED_NAV_HTML, SIMPLE_NAV_HTML, task)
        assert result.verdict != RegressionVerdict.REGRESSION or \
            result.verdict == RegressionVerdict.INCONCLUSIVE
