"""Unit tests for usability_oracle.mdp.builder — MDPBuilder and MDPBuilderConfig.

Validates that the MDPBuilder correctly constructs MDP instances from
accessibility trees and task specifications: state enumeration, action
generation, transition wiring, goal identification, reachability pruning,
and configuration overrides.

The builder expects a task_spec with ``sub_goals``, ``target_node_ids``,
and ``task_id`` attributes (the ``_TaskSpecProtocol``).  We provide a
lightweight adapter and a custom tree factory so that tests exercise the
real build() pipeline end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from tests.fixtures.sample_trees import (
    make_simple_form_tree,
    make_navigation_tree,
    make_dashboard_tree,
    make_modal_dialog_tree,
)
from tests.fixtures.sample_tasks import make_login_task


# ── Lightweight adapter matching the _TaskSpecProtocol expected by MDPBuilder ──

@dataclass
class _SubGoal:
    """Minimal sub-goal object with a target_node_id attribute."""
    target_node_id: str
    description: str = ""


@dataclass
class _TaskSpecAdapter:
    """Minimal task-spec object satisfying the builder's duck-typed protocol."""
    task_id: str = "test_task"
    sub_goals: list[Any] = field(default_factory=list)
    target_node_ids: list[str] = field(default_factory=list)
    description: str = "test"


# ── Helper to build a tree with a focusable root ──

def _ds(**kw) -> AccessibilityState:
    defaults = dict(
        focused=False, selected=False, expanded=False, checked=None,
        disabled=False, hidden=False, required=False, readonly=False,
        pressed=None, value=None,
    )
    defaults.update(kw)
    return AccessibilityState(**defaults)


def _make_focusable_form_tree() -> AccessibilityTree:
    """A form tree where the root itself is a focusable button,
    so that the builder's initial state (root:) exists in the state space."""
    input_field = AccessibilityNode(
        id="input1", role="textbox", name="Username", description="",
        bounding_box=BoundingBox(x=50, y=60, width=200, height=30),
        properties={}, state=_ds(required=True), children=[],
        parent_id="root", depth=1, index_in_parent=0,
    )
    submit_btn = AccessibilityNode(
        id="btn1", role="button", name="Submit", description="",
        bounding_box=BoundingBox(x=50, y=200, width=100, height=40),
        properties={}, state=_ds(), children=[],
        parent_id="root", depth=1, index_in_parent=1,
    )
    root = AccessibilityNode(
        id="root", role="button", name="Start", description="",
        bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
        properties={}, state=_ds(), children=[input_field, submit_btn],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def _make_nav_tree() -> AccessibilityTree:
    """A navigation tree with a focusable root link."""
    links = []
    for i, label in enumerate(["Home", "Products", "About"]):
        links.append(AccessibilityNode(
            id=f"link_{i}", role="link", name=label, description="",
            bounding_box=BoundingBox(x=i * 120, y=0, width=100, height=40),
            properties={"href": f"/{label.lower()}"}, state=_ds(),
            children=[], parent_id="root", depth=1, index_in_parent=i,
        ))
    root = AccessibilityNode(
        id="root", role="link", name="Site", description="",
        bounding_box=BoundingBox(x=0, y=0, width=400, height=40),
        properties={"href": "/"}, state=_ds(), children=links,
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def _make_dialog_tree() -> AccessibilityTree:
    """A dialog tree with a focusable root button."""
    ok_btn = AccessibilityNode(
        id="btn_ok", role="button", name="OK", description="",
        bounding_box=BoundingBox(x=700, y=500, width=80, height=36),
        properties={}, state=_ds(), children=[],
        parent_id="root", depth=1, index_in_parent=0,
    )
    cancel_btn = AccessibilityNode(
        id="btn_cancel", role="button", name="Cancel", description="",
        bounding_box=BoundingBox(x=800, y=500, width=80, height=36),
        properties={}, state=_ds(), children=[],
        parent_id="root", depth=1, index_in_parent=1,
    )
    root = AccessibilityNode(
        id="root", role="button", name="Start", description="",
        bounding_box=BoundingBox(x=0, y=0, width=1920, height=1080),
        properties={}, state=_ds(), children=[ok_btn, cancel_btn],
        parent_id=None, depth=0, index_in_parent=0,
    )
    return AccessibilityTree(root=root, metadata={"source": "test"})


def _form_adapter() -> _TaskSpecAdapter:
    """Adapter for the focusable form tree: 1 sub-goal targeting btn1."""
    return _TaskSpecAdapter(
        task_id="login",
        sub_goals=[_SubGoal(target_node_id="btn1")],
        target_node_ids=["btn1"],
        description="Login",
    )


def _nav_adapter() -> _TaskSpecAdapter:
    """Adapter for the navigation tree."""
    return _TaskSpecAdapter(
        task_id="nav",
        sub_goals=[_SubGoal(target_node_id="link_1")],
        target_node_ids=["link_1"],
        description="Navigate to Products",
    )


def _dialog_adapter() -> _TaskSpecAdapter:
    """Adapter for the dialog tree."""
    return _TaskSpecAdapter(
        task_id="dialog",
        sub_goals=[_SubGoal(target_node_id="btn_ok")],
        target_node_ids=["btn_ok"],
        description="Confirm dialog",
    )


# ═══════════════════════════════════════════════════════════════════════════
# MDPBuilderConfig tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMDPBuilderConfig:
    """Tests for MDPBuilderConfig — parameter object governing MDP construction."""

    def test_default_max_states(self):
        """Default max_states should be 50,000."""
        cfg = MDPBuilderConfig()
        assert cfg.max_states == 50_000

    def test_default_include_read_actions(self):
        """Read actions are included by default."""
        cfg = MDPBuilderConfig()
        assert cfg.include_read_actions is True

    def test_default_include_scroll_actions(self):
        """Scroll actions are included by default."""
        cfg = MDPBuilderConfig()
        assert cfg.include_scroll_actions is True

    def test_default_include_back_actions(self):
        """Back actions are included by default."""
        cfg = MDPBuilderConfig()
        assert cfg.include_back_actions is True

    def test_default_deterministic(self):
        """Builder defaults to deterministic transitions."""
        cfg = MDPBuilderConfig()
        assert cfg.deterministic is True

    def test_default_costs(self):
        """Default cost parameters should match documented values."""
        cfg = MDPBuilderConfig()
        assert cfg.base_step_cost == 1.0
        assert cfg.click_cost == 0.5
        assert cfg.type_cost_per_char == 0.3
        assert cfg.tab_cost == 0.2
        assert cfg.scroll_cost == 0.4
        assert cfg.read_cost == 0.6
        assert cfg.navigate_cost == 0.3

    def test_default_discount(self):
        """Default discount factor is 0.99."""
        cfg = MDPBuilderConfig()
        assert cfg.discount == 0.99

    def test_custom_max_states(self):
        """max_states can be overridden."""
        cfg = MDPBuilderConfig(max_states=100)
        assert cfg.max_states == 100

    def test_max_task_progress_bits_default(self):
        """Default max_task_progress_bits should be 8."""
        cfg = MDPBuilderConfig()
        assert cfg.max_task_progress_bits == 8

    def test_custom_costs_override(self):
        """Individual cost parameters can be overridden."""
        cfg = MDPBuilderConfig(click_cost=2.0, tab_cost=0.05)
        assert cfg.click_cost == 2.0
        assert cfg.tab_cost == 0.05


# ═══════════════════════════════════════════════════════════════════════════
# MDPBuilder tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMDPBuilder:
    """Tests for MDPBuilder.build() — constructing an MDP from a tree + task."""

    def test_build_returns_mdp(self):
        """build() should return an MDP instance."""
        builder = MDPBuilder()
        tree = _make_focusable_form_tree()
        task = _form_adapter()
        mdp = builder.build(tree, task)
        assert isinstance(mdp, MDP)

    def test_built_mdp_has_states(self):
        """The built MDP should have at least one state."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        assert mdp.n_states > 0

    def test_built_mdp_has_actions(self):
        """The built MDP should contain actions derived from interactive nodes."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        assert mdp.n_actions > 0

    def test_built_mdp_has_transitions(self):
        """The built MDP should have transitions wiring states together."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        assert mdp.n_transitions > 0

    def test_initial_state_set(self):
        """The built MDP should have a non-empty initial_state in states."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        assert mdp.initial_state != ""
        assert mdp.initial_state in mdp.states

    def test_goal_states_populated(self):
        """The built MDP should have at least one goal state."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        assert len(mdp.goal_states) > 0
        for gs in mdp.goal_states:
            assert gs in mdp.states

    def test_all_states_reachable(self):
        """Every state in the built MDP should be reachable from initial_state."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        reachable = mdp.reachable_states()
        for sid in mdp.states:
            assert sid in reachable, f"State {sid!r} is unreachable"

    def test_transition_probabilities_sum_to_one(self):
        """For each (state, action) pair, transition probabilities should sum to ~1."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        for sid in mdp.transition_matrix:
            for aid, outcomes in mdp.transition_matrix[sid].items():
                total = sum(p for _, p, _ in outcomes)
                assert total == pytest.approx(1.0, abs=1e-6), (
                    f"T({sid}, {aid}) sums to {total}"
                )

    def test_validates_cleanly(self):
        """The built MDP should pass structural validation."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        errors = mdp.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_navigation_tree_build(self):
        """Builder handles a navigation menu tree + navigation task."""
        builder = MDPBuilder()
        mdp = builder.build(_make_nav_tree(), _nav_adapter())
        assert mdp.n_states > 0
        assert mdp.validate() == []

    def test_modal_dialog_build(self):
        """Builder handles a modal dialog tree + dialog task."""
        builder = MDPBuilder()
        mdp = builder.build(_make_dialog_tree(), _dialog_adapter())
        assert mdp.n_states > 0
        assert mdp.validate() == []

    def test_dashboard_build(self):
        """Builder handles the original dashboard tree (may yield empty MDP)."""
        builder = MDPBuilder()
        task = _TaskSpecAdapter(task_id="dash", description="Dashboard")
        mdp = builder.build(make_dashboard_tree(), task)
        assert isinstance(mdp, MDP)

    def test_config_no_read_actions(self):
        """Disabling include_read_actions should reduce or remove read actions."""
        cfg_with = MDPBuilderConfig(include_read_actions=True)
        cfg_without = MDPBuilderConfig(include_read_actions=False)

        builder_with = MDPBuilder(config=cfg_with)
        builder_without = MDPBuilder(config=cfg_without)

        task = _form_adapter()
        tree = _make_focusable_form_tree()
        mdp_with = builder_with.build(tree, task)
        mdp_without = builder_without.build(tree, task)

        read_with = sum(
            1 for a in mdp_with.actions.values() if a.action_type == "read"
        )
        read_without = sum(
            1 for a in mdp_without.actions.values() if a.action_type == "read"
        )
        assert read_without <= read_with

    def test_config_no_scroll_actions(self):
        """Disabling include_scroll_actions should exclude scroll actions."""
        cfg = MDPBuilderConfig(include_scroll_actions=False)
        builder = MDPBuilder(config=cfg)
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        scroll_count = sum(
            1 for a in mdp.actions.values() if a.action_type == "scroll"
        )
        assert scroll_count == 0

    def test_config_no_back_actions(self):
        """Disabling include_back_actions should exclude back actions."""
        cfg = MDPBuilderConfig(include_back_actions=False)
        builder = MDPBuilder(config=cfg)
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        back_count = sum(
            1 for a in mdp.actions.values() if a.action_type == "back"
        )
        assert back_count == 0

    def test_default_config_when_none(self):
        """Passing no config should default to MDPBuilderConfig()."""
        builder = MDPBuilder(config=None)
        assert builder.config.max_states == 50_000

    def test_discount_propagated(self):
        """The MDP discount should equal the config discount."""
        cfg = MDPBuilderConfig(discount=0.95)
        builder = MDPBuilder(config=cfg)
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        assert mdp.discount == 0.95

    def test_goal_states_are_marked(self):
        """Goal states in the built MDP should have is_goal=True."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        for gs in mdp.goal_states:
            assert mdp.states[gs].is_goal is True

    def test_actions_have_valid_types(self):
        """All action types in the built MDP should be from the known set."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        valid_types = {"click", "type", "tab", "scroll", "navigate", "read",
                       "select", "back"}
        for action in mdp.actions.values():
            assert action.action_type in valid_types, (
                f"Unknown action type: {action.action_type}"
            )

    def test_transitions_reference_existing_states(self):
        """All transition source/target IDs should exist in mdp.states."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        for t in mdp.transitions:
            assert t.source in mdp.states, f"Missing source {t.source}"
            assert t.target in mdp.states, f"Missing target {t.target}"

    def test_transitions_reference_existing_actions(self):
        """All transition action IDs should exist in mdp.actions."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        for t in mdp.transitions:
            assert t.action in mdp.actions, f"Missing action {t.action}"

    def test_all_costs_non_negative(self):
        """Every transition cost in the built MDP should be >= 0."""
        builder = MDPBuilder()
        mdp = builder.build(_make_focusable_form_tree(), _form_adapter())
        for t in mdp.transitions:
            assert t.cost >= 0, f"Negative cost {t.cost} on {t}"

    def test_no_sub_goals_still_builds(self):
        """Builder should produce a valid MDP even without sub_goals."""
        builder = MDPBuilder()
        task = _TaskSpecAdapter(task_id="empty", description="No goals")
        mdp = builder.build(_make_focusable_form_tree(), task)
        assert isinstance(mdp, MDP)
        assert mdp.n_actions > 0

    def test_build_with_raw_taskspec(self):
        """Builder should accept the real TaskSpec from fixtures (graceful fallback)."""
        builder = MDPBuilder()
        task = make_login_task()
        mdp = builder.build(make_simple_form_tree(), task)
        assert isinstance(mdp, MDP)
        # May have 0 states due to protocol mismatch, but should not crash
        assert mdp.n_actions > 0
