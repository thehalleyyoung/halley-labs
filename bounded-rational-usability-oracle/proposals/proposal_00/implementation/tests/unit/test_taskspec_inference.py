"""Tests for usability_oracle.taskspec.inference — TaskInferrer.

This module verifies that the TaskInferrer correctly detects common UI
patterns (forms, navigation, search, selection, dialogs) from mock
accessibility trees and emits well-formed TaskSpec candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import pytest

from usability_oracle.taskspec.inference import TaskInferrer
from usability_oracle.taskspec.models import TaskSpec


# ===================================================================
# Mock accessibility-tree infrastructure
# ===================================================================


@dataclass
class MockNode:
    """Minimal implementation of the AccessibilityNode protocol."""
    node_id: str = ""
    role: str = ""
    name: str = ""
    children: List["MockNode"] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockTree:
    """Minimal implementation of the AccessibilityTree protocol."""
    root: MockNode = field(default_factory=MockNode)

    def find_by_role(self, role: str) -> List[MockNode]:
        """Breadth-first search for nodes matching *role*."""
        results: List[MockNode] = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.role.lower() == role.lower():
                results.append(node)
            queue.extend(node.children)
        return results

    def find_by_name(self, name: str) -> List[MockNode]:
        results: List[MockNode] = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.name == name:
                results.append(node)
            queue.extend(node.children)
        return results


# ===================================================================
# Tree builders
# ===================================================================


def _form_tree(num_fields: int = 3) -> MockTree:
    """Build a tree with a form containing *num_fields* text inputs and
    a submit button."""
    inputs = [
        MockNode(node_id=f"inp_{i}", role="textfield", name=f"Field{i}")
        for i in range(num_fields)
    ]
    submit = MockNode(node_id="btn_submit", role="button", name="Submit")
    form = MockNode(node_id="form1", role="form", name="MyForm",
                    children=inputs + [submit])
    root = MockNode(node_id="root", role="document", name="Page",
                    children=[form])
    return MockTree(root=root)


def _nav_tree() -> MockTree:
    """Build a tree with a navigation bar containing link children."""
    links = [
        MockNode(node_id=f"link_{i}", role="link", name=name)
        for i, name in enumerate(["Home", "About", "Contact"])
    ]
    nav = MockNode(node_id="nav1", role="navigation", name="MainNav",
                   children=links)
    root = MockNode(node_id="root", role="document", name="Page",
                    children=[nav])
    return MockTree(root=root)


def _search_tree() -> MockTree:
    """Build a tree with a search box."""
    searchbox = MockNode(node_id="sb1", role="searchbox", name="Search")
    root = MockNode(node_id="root", role="document", name="Page",
                    children=[searchbox])
    return MockTree(root=root)


def _selection_tree() -> MockTree:
    """Build a tree with a listbox containing selectable items."""
    items = [
        MockNode(node_id=f"item_{i}", role="listitem", name=f"Item {i}")
        for i in range(5)
    ]
    listbox = MockNode(node_id="lb1", role="listbox", name="Options",
                       children=items)
    root = MockNode(node_id="root", role="document", name="Page",
                    children=[listbox])
    return MockTree(root=root)


def _dialog_tree(has_confirm: bool = True, has_cancel: bool = True,
                 has_inputs: bool = True) -> MockTree:
    """Build a tree with a dialog containing optional inputs and buttons."""
    children: list[MockNode] = []
    if has_inputs:
        children.append(
            MockNode(node_id="dlg_inp", role="textfield", name="Reason"))
    if has_confirm:
        children.append(
            MockNode(node_id="dlg_ok", role="button", name="Confirm"))
    if has_cancel:
        children.append(
            MockNode(node_id="dlg_cancel", role="button", name="Cancel"))
    dialog = MockNode(node_id="dlg1", role="dialog", name="Confirm Action",
                      children=children)
    root = MockNode(node_id="root", role="document", name="Page",
                    children=[dialog])
    return MockTree(root=root)


def _empty_tree() -> MockTree:
    """An empty tree with just a root node."""
    return MockTree(root=MockNode(node_id="root", role="document", name="Page"))


def _no_interactive_tree() -> MockTree:
    """A tree with content but no interactive elements."""
    heading = MockNode(node_id="h1", role="heading", name="Welcome")
    paragraph = MockNode(node_id="p1", role="paragraph", name="Some text")
    root = MockNode(node_id="root", role="document", name="Page",
                    children=[heading, paragraph])
    return MockTree(root=root)


# ===================================================================
# Form detection tests
# ===================================================================


class TestFormDetection:
    """Verify that form-filling task patterns are correctly inferred."""

    def test_form_detected(self):
        """A tree with a form containing inputs and a submit button should
        produce at least one task spec with the 'form' pattern."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_form_tree())
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert len(form_specs) >= 1

    def test_form_spec_has_steps(self):
        """The inferred form spec should contain click+type pairs for each
        input field and a final submit click."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_form_tree(num_fields=3))
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert form_specs
        flow = form_specs[0].flows[0]
        # 3 fields × 2 steps (click+type) + 1 submit = 7
        assert len(flow.steps) == 7

    def test_form_spec_metadata(self):
        """Form specs should have inferred=True and pattern=form metadata."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_form_tree())
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert form_specs[0].metadata["inferred"] is True
        assert form_specs[0].metadata["pattern"] == "form"

    def test_form_spec_is_valid_taskspec(self):
        """Every inferred spec should be a proper TaskSpec instance with
        a name, at least one flow, and steps."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_form_tree())
        for spec in specs:
            assert isinstance(spec, TaskSpec)
            assert spec.name
            assert len(spec.flows) >= 1
            for flow in spec.flows:
                assert len(flow.steps) >= 1

    def test_min_form_fields_enforcement(self):
        """A form with fewer inputs than min_form_fields should not
        produce a form spec."""
        inferrer = TaskInferrer(min_form_fields=5)
        specs = inferrer.infer_from_tree(_form_tree(num_fields=3))
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert len(form_specs) == 0

    def test_form_with_exact_min_fields(self):
        """A form with exactly min_form_fields inputs should be detected."""
        inferrer = TaskInferrer(min_form_fields=2)
        specs = inferrer.infer_from_tree(_form_tree(num_fields=2))
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert len(form_specs) >= 1


# ===================================================================
# Navigation detection tests
# ===================================================================


class TestNavigationDetection:
    """Verify that navigation task patterns are correctly inferred."""

    def test_nav_detected(self):
        """A navigation bar with links should produce nav-pattern specs."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_nav_tree())
        nav_specs = [s for s in specs if s.metadata.get("pattern") == "navigation"]
        assert len(nav_specs) >= 1

    def test_nav_per_link(self):
        """Each link in the nav should generate a separate spec."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_nav_tree())
        nav_specs = [s for s in specs if s.metadata.get("pattern") == "navigation"]
        # 3 links → 3 specs
        assert len(nav_specs) == 3

    def test_nav_spec_steps(self):
        """Each nav spec should have a single click step."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_nav_tree())
        nav_specs = [s for s in specs if s.metadata.get("pattern") == "navigation"]
        for spec in nav_specs:
            assert len(spec.flows[0].steps) == 1
            assert spec.flows[0].steps[0].action_type == "click"

    def test_nav_step_targets(self):
        """Nav step target names should match the link labels."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_nav_tree())
        nav_specs = [s for s in specs if s.metadata.get("pattern") == "navigation"]
        names = {spec.flows[0].steps[0].target_name for spec in nav_specs}
        assert "Home" in names
        assert "About" in names
        assert "Contact" in names


# ===================================================================
# Search detection tests
# ===================================================================


class TestSearchDetection:
    """Verify that search-and-select patterns are correctly inferred."""

    def test_search_detected(self):
        """A searchbox should trigger a search-pattern spec."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_search_tree())
        search_specs = [s for s in specs if s.metadata.get("pattern") == "search"]
        assert len(search_specs) >= 1

    def test_search_flow_steps(self):
        """The search flow should have click → type → wait → click steps."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_search_tree())
        search_specs = [s for s in specs if s.metadata.get("pattern") == "search"]
        flow = search_specs[0].flows[0]
        actions = [s.action_type for s in flow.steps]
        assert actions == ["click", "type", "wait", "click"]

    def test_search_query_placeholder(self):
        """The type step should carry a placeholder input_value."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_search_tree())
        search_specs = [s for s in specs if s.metadata.get("pattern") == "search"]
        type_step = search_specs[0].flows[0].steps[1]
        assert type_step.input_value == "<search_query>"


# ===================================================================
# Selection detection tests
# ===================================================================


class TestSelectionDetection:
    """Verify that selection patterns (list/combobox) are detected."""

    def test_selection_detected(self):
        """A listbox with items should generate a selection spec."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_selection_tree())
        sel_specs = [s for s in specs if s.metadata.get("pattern") == "selection"]
        assert len(sel_specs) >= 1

    def test_selection_item_count_metadata(self):
        """The inferred spec should record the number of selectable items."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_selection_tree())
        sel_specs = [s for s in specs if s.metadata.get("pattern") == "selection"]
        assert sel_specs[0].metadata["item_count"] == 5

    def test_selection_has_optional_scroll(self):
        """The selection flow should include an optional scroll step."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_selection_tree())
        sel_specs = [s for s in specs if s.metadata.get("pattern") == "selection"]
        flow = sel_specs[0].flows[0]
        opt_steps = [s for s in flow.steps if s.optional]
        assert len(opt_steps) >= 1


# ===================================================================
# Dialog detection tests
# ===================================================================


class TestDialogDetection:
    """Verify that dialog interaction patterns are detected."""

    def test_dialog_detected(self):
        """A dialog element should generate a dialog-pattern spec."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_dialog_tree())
        dlg_specs = [s for s in specs if s.metadata.get("pattern") == "dialog"]
        assert len(dlg_specs) >= 1

    def test_dialog_has_confirm_flow(self):
        """The dialog spec should include a confirm flow."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_dialog_tree())
        dlg_specs = [s for s in specs if s.metadata.get("pattern") == "dialog"]
        flow_names = [f.name for f in dlg_specs[0].flows]
        assert any("confirm" in n for n in flow_names)

    def test_dialog_has_cancel_flow(self):
        """When a cancel button exists the spec should include a cancel flow."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_dialog_tree(has_cancel=True))
        dlg_specs = [s for s in specs if s.metadata.get("pattern") == "dialog"]
        flow_names = [f.name for f in dlg_specs[0].flows]
        assert any("cancel" in n for n in flow_names)

    def test_dialog_without_cancel(self):
        """A dialog without a cancel button should have only a confirm flow."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(
            _dialog_tree(has_cancel=False, has_confirm=True))
        dlg_specs = [s for s in specs if s.metadata.get("pattern") == "dialog"]
        flow_names = [f.name for f in dlg_specs[0].flows]
        assert not any("cancel" in n for n in flow_names)

    def test_dialog_metadata(self):
        """Dialog spec metadata should record has_inputs, has_confirm, has_cancel."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_dialog_tree())
        dlg_specs = [s for s in specs if s.metadata.get("pattern") == "dialog"]
        meta = dlg_specs[0].metadata
        assert meta["has_inputs"] is True
        assert meta["has_confirm"] is True
        assert meta["has_cancel"] is True


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Empty tree, no interactive elements, and parameter boundaries."""

    def test_empty_tree_returns_empty(self):
        """An empty tree (root with no children) should produce no specs."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_empty_tree())
        assert specs == []

    def test_no_interactive_elements_returns_empty(self):
        """A tree with only non-interactive content should produce no specs."""
        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(_no_interactive_tree())
        assert specs == []

    def test_min_form_fields_one(self):
        """Setting min_form_fields=1 should detect even single-field forms."""
        inferrer = TaskInferrer(min_form_fields=1)
        specs = inferrer.infer_from_tree(_form_tree(num_fields=1))
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert len(form_specs) >= 1

    def test_multiple_patterns_in_complex_tree(self):
        """A tree containing both a form and a nav bar should yield specs
        for both patterns."""
        form_inputs = [
            MockNode(node_id=f"inp_{i}", role="textfield", name=f"F{i}")
            for i in range(3)
        ]
        submit = MockNode(node_id="btn", role="button", name="Go")
        form = MockNode(node_id="form1", role="form", name="MyForm",
                        children=form_inputs + [submit])
        links = [MockNode(node_id="lnk", role="link", name="Help")]
        nav = MockNode(node_id="nav1", role="navigation", name="Nav",
                       children=links)
        root = MockNode(node_id="root", role="document", children=[form, nav])
        tree = MockTree(root=root)

        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(tree)
        patterns = {s.metadata.get("pattern") for s in specs}
        assert "form" in patterns
        assert "navigation" in patterns

    def test_combobox_in_form_uses_select_action(self):
        """A combobox field in a form should generate a 'select' action
        rather than a 'type' action."""
        combo = MockNode(node_id="combo1", role="combobox", name="Country")
        text = MockNode(node_id="txt1", role="textfield", name="Name")
        submit = MockNode(node_id="btn", role="button", name="OK")
        form = MockNode(node_id="form1", role="form", name="RegForm",
                        children=[combo, text, submit])
        root = MockNode(node_id="root", role="document", children=[form])
        tree = MockTree(root=root)

        inferrer = TaskInferrer()
        specs = inferrer.infer_from_tree(tree)
        form_specs = [s for s in specs if s.metadata.get("pattern") == "form"]
        assert form_specs
        actions = [s.action_type for s in form_specs[0].flows[0].steps]
        assert "select" in actions
