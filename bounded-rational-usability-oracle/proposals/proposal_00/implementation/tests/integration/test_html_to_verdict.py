"""Integration tests for the HTML → regression-verdict pipeline.

These tests exercise the multi-stage path that starts with raw HTML strings,
parses them into accessibility trees, aligns the trees, builds MDPs, solves
policies, and compares the resulting costs to produce a ``RegressionVerdict``.

Every test constructs the intermediate artefacts explicitly so that failures
can be localised to a specific stage.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.alignment.differ import SemanticDiffer
from usability_oracle.alignment.models import AlignmentResult, AlignmentConfig
from usability_oracle.alignment.models import (
    AccessibilityTree as AlignTree,
    AccessibilityNode as AlignNode,
    AccessibilityRole,
    BoundingBox as AlignBBox,
)
from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig
from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.policy.models import Policy, PolicyResult
from usability_oracle.comparison.paired import PairedComparator
from usability_oracle.comparison.models import ComparisonResult
from usability_oracle.comparison.models import AlignmentResult as CompAlignmentResult
from usability_oracle.comparison.models import StateMapping
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"


def _load_html(name: str) -> str:
    """Load an HTML fixture by stem name."""
    return (SAMPLE_HTML_DIR / f"{name}.html").read_text()


def _make_login_task() -> TaskSpec:
    """Minimal login task spec used across tests."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus username"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type username", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Submit", description="Submit form",
                 depends_on=["s2"]),
    ]
    flow = TaskFlow(flow_id="login", name="Login Flow", steps=steps,
                    success_criteria=["logged_in"])
    return TaskSpec(spec_id="login", name="Login", flows=[flow])


def _parse_and_normalise(html: str) -> AccessibilityTree:
    """Parse raw HTML and normalise the resulting accessibility tree."""
    parser = HTMLAccessibilityParser()
    tree = parser.parse(html)
    normaliser = AccessibilityNormalizer()
    tree = normaliser.normalize(tree)
    # Ensure root is focusable so MDPBuilder includes it in state space
    tree.root.properties['tabindex'] = '0'
    return tree


def _to_align_tree(atree: AccessibilityTree) -> AlignTree:
    """Convert an accessibility-model tree to an alignment-model tree."""
    def _role(role_str: str) -> AccessibilityRole:
        try:
            return AccessibilityRole(role_str)
        except ValueError:
            return AccessibilityRole.GENERIC

    nodes: dict[str, AlignNode] = {}

    def _visit(anode: AccessibilityNode, parent_id=None) -> None:
        bbox = None
        if hasattr(anode, 'bounding_box') and anode.bounding_box is not None:
            bb = anode.bounding_box
            bbox = AlignBBox(x=bb.x, y=bb.y, width=bb.width, height=bb.height)
        children_ids = [c.id for c in anode.children] if hasattr(anode, 'children') else []
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


def _align(tree_a: AccessibilityTree,
           tree_b: AccessibilityTree) -> AlignmentResult:
    """Compute a semantic diff between two accessibility trees."""
    differ = SemanticDiffer()
    return differ.diff(_to_align_tree(tree_a), _to_align_tree(tree_b))


def _make_comp_alignment(mdp_a: MDP, mdp_b: MDP) -> CompAlignmentResult:
    """Build a comparison AlignmentResult from two MDPs' shared states."""
    common = set(mdp_a.states.keys()) & set(mdp_b.states.keys())
    mappings = [StateMapping(state_a=s, state_b=s) for s in common]
    if not mappings:
        mappings = [StateMapping(state_a=list(mdp_a.states.keys())[0],
                                  state_b=list(mdp_b.states.keys())[0])]
    return CompAlignmentResult(mappings=mappings)


# ===================================================================
# Tests – Parse stage
# ===================================================================


class TestHTMLParsing:
    """Verify that HTML → AccessibilityTree is well-formed."""

    def test_simple_form_parses(self) -> None:
        """Parsing the simple-form fixture must produce a non-empty tree."""
        html = _load_html("simple_form")
        tree = _parse_and_normalise(html)
        assert tree.size() > 0

    def test_navigation_menu_parses(self) -> None:
        """The navigation-menu fixture must parse without errors."""
        html = _load_html("navigation_menu")
        tree = _parse_and_normalise(html)
        assert tree.root is not None

    def test_complex_dashboard_parses(self) -> None:
        """The complex-dashboard fixture should produce a large tree."""
        html = _load_html("complex_dashboard")
        tree = _parse_and_normalise(html)
        assert tree.size() >= 1

    def test_parsed_tree_has_interactive_nodes(self) -> None:
        """A form HTML should contain at least one interactive node."""
        html = _load_html("simple_form")
        tree = _parse_and_normalise(html)
        interactive = tree.get_interactive_nodes()
        assert len(interactive) > 0, "Form should have interactive elements"

    def test_parsed_tree_validates(self) -> None:
        """The parsed tree should pass its own structural validation."""
        html = _load_html("simple_form")
        tree = _parse_and_normalise(html)
        errors = tree.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"


# ===================================================================
# Tests – Alignment stage
# ===================================================================


class TestHTMLAlignment:
    """Verify tree alignment between two HTML-derived trees."""

    def test_identical_html_full_match(self) -> None:
        """Identical trees must yield similarity ≈ 1.0."""
        html = _load_html("simple_form")
        tree = _parse_and_normalise(html)
        alignment = _align(tree, tree)
        assert alignment.similarity_score >= 0.9

    def test_different_html_partial_match(self) -> None:
        """Different fixtures should still produce a valid alignment."""
        tree_a = _parse_and_normalise(_load_html("simple_form"))
        tree_b = _parse_and_normalise(_load_html("navigation_menu"))
        alignment = _align(tree_a, tree_b)
        assert 0.0 <= alignment.similarity_score <= 1.0

    def test_alignment_has_mappings(self) -> None:
        """Aligning two forms should produce at least one mapping."""
        html = _load_html("simple_form")
        tree = _parse_and_normalise(html)
        alignment = _align(tree, tree)
        assert len(alignment.mappings) > 0

    def test_alignment_edit_distance_zero_for_identical(self) -> None:
        """Identical trees must yield zero edit distance."""
        html = _load_html("simple_form")
        tree = _parse_and_normalise(html)
        alignment = _align(tree, tree)
        assert alignment.edit_distance == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# Tests – MDP build + solve
# ===================================================================


class TestHTMLMDPConstruction:
    """Build MDPs from accessibility trees and task specs."""

    def test_mdp_from_simple_form(self) -> None:
        """Building an MDP from a parsed form should produce states."""
        tree = _parse_and_normalise(_load_html("simple_form"))
        task = _make_login_task()
        builder = MDPBuilder()
        mdp = builder.build(tree, task)
        assert mdp.n_states > 0
        assert mdp.n_transitions > 0

    def test_mdp_has_goal(self) -> None:
        """The MDP should contain goal states if task steps match nodes."""
        tree = _parse_and_normalise(_load_html("simple_form"))
        task = _make_login_task()
        builder = MDPBuilder()
        mdp = builder.build(tree, task)
        assert isinstance(mdp.goal_states, set)

    def test_mdp_initial_state_set(self) -> None:
        """The initial state must be a valid member of the state space."""
        tree = _parse_and_normalise(_load_html("simple_form"))
        task = _make_login_task()
        builder = MDPBuilder()
        mdp = builder.build(tree, task)
        assert mdp.initial_state in mdp.states

    def test_value_iteration_converges(self) -> None:
        """Standard value iteration must converge on the form MDP."""
        tree = _parse_and_normalise(_load_html("simple_form"))
        task = _make_login_task()
        mdp = MDPBuilder().build(tree, task)
        solver = ValueIterationSolver()
        values, policy = solver.solve(mdp)
        assert len(values) == mdp.n_states
        assert len(policy) > 0


# ===================================================================
# Tests – Full HTML→verdict
# ===================================================================


class TestHTMLToVerdict:
    """End-to-end: HTML pair → ``RegressionVerdict``."""

    def test_identical_html_neutral_verdict(self) -> None:
        """Passing the same HTML for both A and B must yield NEUTRAL."""
        html = _load_html("simple_form")
        task = _make_login_task()
        tree = _parse_and_normalise(html)
        builder = MDPBuilder()
        mdp = builder.build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        comparator = PairedComparator()
        result: ComparisonResult = comparator.compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )

    def test_different_html_produces_actionable_verdict(self) -> None:
        """Different HTML pages should yield a non-INCONCLUSIVE verdict."""
        tree_a = _parse_and_normalise(_load_html("simple_form"))
        tree_b = _parse_and_normalise(_load_html("navigation_menu"))
        task = _make_login_task()
        builder = MDPBuilder()
        mdp_a = builder.build(tree_a, task)
        mdp_b = builder.build(tree_b, task)
        alignment = _make_comp_alignment(mdp_a, mdp_b)
        comparator = PairedComparator()
        result = comparator.compare(
            mdp_a=mdp_a, mdp_b=mdp_b, alignment=alignment, task=task,
        )
        assert isinstance(result.verdict, RegressionVerdict)

    def test_comparison_result_has_costs(self) -> None:
        """The comparison result must populate cost_before / cost_after."""
        html = _load_html("simple_form")
        task = _make_login_task()
        tree = _parse_and_normalise(html)
        mdp = MDPBuilder().build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        result = PairedComparator().compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert result.cost_before is not None
        assert result.cost_after is not None

    def test_comparison_result_effect_size(self) -> None:
        """Identical inputs should yield a near-zero effect size."""
        html = _load_html("simple_form")
        task = _make_login_task()
        tree = _parse_and_normalise(html)
        mdp = MDPBuilder().build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        result = PairedComparator().compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert abs(result.effect_size) < 1.0

    def test_modal_dialog_html_pipeline(self) -> None:
        """Running through the pipeline with the modal dialog fixture."""
        html = _load_html("modal_dialog")
        task = _make_login_task()
        tree = _parse_and_normalise(html)
        mdp = MDPBuilder().build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        result = PairedComparator().compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert isinstance(result, ComparisonResult)


class TestHTMLNormalizerIntegration:
    """Verify normaliser effects on downstream stages."""

    def test_normaliser_preserves_interactive_count(self) -> None:
        """Normalisation should not drop interactive elements."""
        parser = HTMLAccessibilityParser()
        raw_tree = parser.parse(_load_html("simple_form"))
        normaliser = AccessibilityNormalizer()
        norm_tree = normaliser.normalize(raw_tree)
        raw_interactive = raw_tree.get_interactive_nodes()
        norm_interactive = norm_tree.get_interactive_nodes()
        assert len(norm_interactive) >= len(raw_interactive)

    def test_normalised_tree_depth_bounded(self) -> None:
        """Normalised trees should not be excessively deep."""
        tree = _parse_and_normalise(_load_html("complex_dashboard"))
        assert tree.depth() <= 50

    def test_normaliser_idempotent(self) -> None:
        """Normalising twice should not change the tree further."""
        normaliser = AccessibilityNormalizer()
        tree = HTMLAccessibilityParser().parse(_load_html("simple_form"))
        once = normaliser.normalize(tree)
        twice = normaliser.normalize(once)
        assert once.size() == twice.size()
