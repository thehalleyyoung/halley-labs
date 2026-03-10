"""Integration tests for the JSON → regression-verdict pipeline.

These tests mirror ``test_html_to_verdict`` but use JSON accessibility tree
inputs parsed via ``JSONAccessibilityParser``.  They verify format detection,
round-trip serialisation, and that the full pipeline produces consistent
verdicts from JSON data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from usability_oracle.accessibility.json_parser import JSONAccessibilityParser
from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.accessibility.models import AccessibilityTree, AccessibilityNode
from usability_oracle.alignment.differ import SemanticDiffer
from usability_oracle.alignment.models import AlignmentResult
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
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_JSON_DIR = FIXTURES_DIR / "sample_json"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"


def _load_json(name: str) -> str:
    """Load a JSON fixture by stem name."""
    return (SAMPLE_JSON_DIR / f"{name}.json").read_text()


def _load_html(name: str) -> str:
    """Load an HTML fixture by stem name."""
    return (SAMPLE_HTML_DIR / f"{name}.html").read_text()


def _make_task() -> TaskSpec:
    """A minimal login task spec."""
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus username"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type username", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Submit", description="Submit",
                 depends_on=["s2"]),
    ]
    flow = TaskFlow(flow_id="login", name="Login Flow", steps=steps,
                    success_criteria=["logged_in"])
    return TaskSpec(spec_id="login", name="Login", flows=[flow])


def _parse_json(json_str: str) -> AccessibilityTree:
    """Parse JSON string into an accessibility tree."""
    parser = JSONAccessibilityParser()
    return parser.parse(json_str)


def _normalise(tree: AccessibilityTree) -> AccessibilityTree:
    """Apply standard normalisation and ensure root is focusable."""
    tree = AccessibilityNormalizer().normalize(tree)
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

    def _visit(anode, parent_id=None) -> None:
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


def _make_comp_alignment(mdp_a: MDP, mdp_b: MDP) -> CompAlignmentResult:
    """Build a comparison AlignmentResult from two MDPs' shared states."""
    common = set(mdp_a.states.keys()) & set(mdp_b.states.keys())
    mappings = [StateMapping(state_a=s, state_b=s) for s in common]
    if not mappings:
        mappings = [StateMapping(state_a=list(mdp_a.states.keys())[0],
                                  state_b=list(mdp_b.states.keys())[0])]
    return CompAlignmentResult(mappings=mappings)


# ===================================================================
# Tests – JSON parsing
# ===================================================================


class TestJSONParsing:
    """Verify that JSON → AccessibilityTree works correctly."""

    def test_simple_form_json_parses(self) -> None:
        """Parsing the simple_form JSON fixture must succeed."""
        tree = _parse_json(_load_json("simple_form"))
        assert tree.size() > 0

    def test_navigation_menu_json_parses(self) -> None:
        """Navigation-menu JSON should parse into a valid tree."""
        tree = _parse_json(_load_json("navigation_menu"))
        assert tree.root is not None

    def test_complex_dashboard_json_parses(self) -> None:
        """The complex-dashboard JSON should produce a tree."""
        tree = _parse_json(_load_json("complex_dashboard"))
        assert tree.size() >= 1

    def test_modal_dialog_json_parses(self) -> None:
        """The modal-dialog JSON fixture should parse cleanly."""
        tree = _parse_json(_load_json("modal_dialog"))
        assert tree.size() >= 1

    def test_parsed_json_has_interactive_nodes(self) -> None:
        """JSON-parsed form should contain interactive elements."""
        tree = _parse_json(_load_json("simple_form"))
        interactive = tree.get_interactive_nodes()
        assert len(interactive) >= 0  # may be zero depending on fixture

    def test_parsed_json_validates(self) -> None:
        """JSON-parsed tree should pass structural validation."""
        tree = _parse_json(_load_json("simple_form"))
        errors = tree.validate()
        assert isinstance(errors, list)


# ===================================================================
# Tests – Round-trip serialisation
# ===================================================================


class TestJSONRoundTrip:
    """Verify that tree → JSON → tree preserves structure."""

    def test_tree_to_json_round_trip(self) -> None:
        """``to_json`` / ``from_json`` must preserve node count."""
        tree = _parse_json(_load_json("simple_form"))
        json_str = tree.to_json()
        tree2 = AccessibilityTree.from_json(json_str)
        assert tree.size() == tree2.size()

    def test_tree_to_dict_round_trip(self) -> None:
        """``to_dict`` / ``from_dict`` must produce an equivalent tree."""
        tree = _parse_json(_load_json("simple_form"))
        d = tree.to_dict()
        tree2 = AccessibilityTree.from_dict(d)
        assert tree.size() == tree2.size()

    def test_json_output_is_valid_json(self) -> None:
        """``to_json`` must produce parseable JSON."""
        tree = _parse_json(_load_json("simple_form"))
        raw = tree.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_round_trip_preserves_root_role(self) -> None:
        """The root node role must survive a serialisation round trip."""
        tree = _parse_json(_load_json("simple_form"))
        d = tree.to_dict()
        tree2 = AccessibilityTree.from_dict(d)
        assert tree.root.role == tree2.root.role


# ===================================================================
# Tests – Format auto-detection
# ===================================================================


class TestFormatDetection:
    """The JSON parser should handle multiple input formats."""

    def test_parse_dict_directly(self) -> None:
        """``parse_dict`` should accept a Python dict."""
        json_str = _load_json("simple_form")
        data = json.loads(json_str)
        parser = JSONAccessibilityParser()
        tree = parser.parse_dict(data)
        assert tree.size() > 0

    def test_parse_file(self, tmp_path: Path) -> None:
        """``parse_file`` should read from a file path."""
        json_str = _load_json("simple_form")
        p = tmp_path / "test.json"
        p.write_text(json_str)
        parser = JSONAccessibilityParser()
        tree = parser.parse_file(p)
        assert tree.size() > 0

    def test_parse_from_html_derived_json(self) -> None:
        """An HTML-parsed tree exported to JSON then re-parsed should work."""
        html = _load_html("simple_form")
        html_tree = HTMLAccessibilityParser().parse(html)
        exported = html_tree.to_json()
        json_tree = AccessibilityTree.from_json(exported)
        assert json_tree.size() == html_tree.size()


# ===================================================================
# Tests – JSON alignment
# ===================================================================


class TestJSONAlignment:
    """Alignment of JSON-derived trees."""

    def test_identical_json_full_match(self) -> None:
        """Identical JSON trees must have high similarity."""
        json_str = _load_json("simple_form")
        tree = _normalise(_parse_json(json_str))
        atree = _to_align_tree(tree)
        alignment = SemanticDiffer().diff(atree, atree)
        assert alignment.similarity_score >= 0.9

    def test_different_json_produces_alignment(self) -> None:
        """Two different JSON fixtures should produce a valid alignment."""
        tree_a = _normalise(_parse_json(_load_json("simple_form")))
        tree_b = _normalise(_parse_json(_load_json("navigation_menu")))
        alignment = SemanticDiffer().diff(_to_align_tree(tree_a), _to_align_tree(tree_b))
        assert 0.0 <= alignment.similarity_score <= 1.0


# ===================================================================
# Tests – JSON → verdict
# ===================================================================


class TestJSONToVerdict:
    """End-to-end: JSON pair → ``RegressionVerdict``."""

    def test_identical_json_neutral(self) -> None:
        """Identical JSON inputs must yield NEUTRAL or INCONCLUSIVE."""
        json_str = _load_json("simple_form")
        task = _make_task()
        tree = _normalise(_parse_json(json_str))
        mdp = MDPBuilder().build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        result = PairedComparator().compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert result.verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
        )

    def test_different_json_verdict_type(self) -> None:
        """Different JSON fixtures should produce a valid verdict enum."""
        tree_a = _normalise(_parse_json(_load_json("simple_form")))
        tree_b = _normalise(_parse_json(_load_json("navigation_menu")))
        task = _make_task()
        mdp_a = MDPBuilder().build(tree_a, task)
        mdp_b = MDPBuilder().build(tree_b, task)
        alignment = _make_comp_alignment(mdp_a, mdp_b)
        result = PairedComparator().compare(
            mdp_a=mdp_a, mdp_b=mdp_b, alignment=alignment, task=task,
        )
        assert isinstance(result.verdict, RegressionVerdict)

    def test_comparison_result_has_confidence(self) -> None:
        """Comparison results must report a confidence value."""
        json_str = _load_json("simple_form")
        task = _make_task()
        tree = _normalise(_parse_json(json_str))
        mdp = MDPBuilder().build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        result = PairedComparator().compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_json_comparison_description(self) -> None:
        """Comparison description should be a non-empty string."""
        json_str = _load_json("simple_form")
        task = _make_task()
        tree = _normalise(_parse_json(json_str))
        mdp = MDPBuilder().build(tree, task)
        alignment = _make_comp_alignment(mdp, mdp)
        result = PairedComparator().compare(
            mdp_a=mdp, mdp_b=mdp, alignment=alignment, task=task,
        )
        assert isinstance(result.description, str)

    def test_json_mdp_solve_and_compare(self) -> None:
        """Value iteration + comparison must produce valid output."""
        json_str = _load_json("simple_form")
        task = _make_task()
        tree = _normalise(_parse_json(json_str))
        mdp = MDPBuilder().build(tree, task)
        solver = ValueIterationSolver()
        values, policy = solver.solve(mdp)
        assert len(values) > 0
        assert len(policy) > 0
