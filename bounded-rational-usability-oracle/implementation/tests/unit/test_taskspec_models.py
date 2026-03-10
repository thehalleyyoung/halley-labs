"""Tests for usability_oracle.taskspec.models — TaskStep, TaskFlow, TaskSpec, TaskGraph.

This module exercises the core data-model classes that underpin every
task-specification workflow in the bounded-rational usability oracle.
Coverage spans construction, field access, serialisation round-trips,
query helpers, validation, and the dependency-DAG utilities exposed by
TaskGraph.
"""

from __future__ import annotations

import copy
import pytest
from collections import defaultdict

from usability_oracle.taskspec.models import (
    ACTION_TYPES,
    TaskFlow,
    TaskGraph,
    TaskSpec,
    TaskStep,
)


# ===================================================================
# Fixtures
# ===================================================================


def _make_step(
    step_id: str = "s1",
    action_type: str = "click",
    target_role: str = "button",
    target_name: str = "Submit",
    optional: bool = False,
    depends_on: list[str] | None = None,
    input_value: str | None = None,
    **kwargs,
) -> TaskStep:
    """Helper to build a TaskStep with sensible defaults."""
    return TaskStep(
        step_id=step_id,
        action_type=action_type,
        target_role=target_role,
        target_name=target_name,
        optional=optional,
        depends_on=depends_on or [],
        input_value=input_value,
        **kwargs,
    )


def _make_flow(
    flow_id: str = "f1",
    name: str = "main_flow",
    steps: list[TaskStep] | None = None,
    **kwargs,
) -> TaskFlow:
    return TaskFlow(flow_id=flow_id, name=name, steps=steps or [], **kwargs)


def _login_spec() -> TaskSpec:
    """Builds a realistic login-form TaskSpec with two flows."""
    click_user = _make_step("click_user", "click", "textfield", "Username")
    type_user = _make_step("type_user", "type", "textfield", "Username",
                           input_value="alice", depends_on=["click_user"])
    click_pass = _make_step("click_pass", "click", "textfield", "Password",
                            depends_on=["type_user"])
    type_pass = _make_step("type_pass", "type", "textfield", "Password",
                           input_value="secret", depends_on=["click_pass"])
    click_submit = _make_step("click_submit", "click", "button", "Sign In",
                              depends_on=["type_pass"])

    main_flow = _make_flow(
        "login_flow",
        "standard_login",
        [click_user, type_user, click_pass, type_pass, click_submit],
        success_criteria=["authenticated"],
        max_time=30.0,
    )

    optional_step = _make_step("remember", "click", "checkbox",
                               "Remember Me", optional=True)
    alt_flow = _make_flow("alt_flow", "quick_login", [optional_step])

    return TaskSpec(
        spec_id="spec_login",
        name="login_form",
        description="Standard username/password login",
        flows=[main_flow, alt_flow],
        initial_state={"page": "/login"},
        metadata={"author": "test"},
    )


# ===================================================================
# TaskStep tests
# ===================================================================


class TestTaskStepFields:
    """Verify that TaskStep stores and exposes all documented fields."""

    def test_default_fields(self):
        """A TaskStep created with only step_id and action_type should
        carry sensible defaults for every other field."""
        step = _make_step()
        assert step.step_id == "s1"
        assert step.action_type == "click"
        assert step.target_role == "button"
        assert step.target_name == "Submit"
        assert step.optional is False
        assert step.depends_on == []
        assert step.input_value is None
        assert step.preconditions == []
        assert step.postconditions == []
        assert step.timeout is None
        assert step.metadata == {}

    def test_auto_generated_step_id(self):
        """Omitting step_id should produce a non-empty auto-generated ID."""
        step = TaskStep(action_type="click")
        assert step.step_id.startswith("step-")
        assert len(step.step_id) > 5

    def test_invalid_action_type_raises(self):
        """Creating a step with an unrecognised action_type must raise
        ValueError so that typos surface immediately."""
        with pytest.raises(ValueError, match="Unknown action_type"):
            TaskStep(step_id="bad", action_type="destroy")


class TestTaskStepProperties:
    """Test computed properties and helpers on TaskStep."""

    def test_is_input_action_type_and_select(self):
        """'type' and 'select' are input actions; 'click' is not."""
        assert _make_step(action_type="type", input_value="hello").is_input_action is True
        assert _make_step(action_type="select", input_value="opt").is_input_action is True
        assert _make_step(action_type="click").is_input_action is False

    def test_target_descriptor_with_role_and_name(self):
        """target_descriptor should join role and quoted name."""
        step = _make_step(target_role="button", target_name="Submit")
        assert "button" in step.target_descriptor
        assert '"Submit"' in step.target_descriptor

    def test_target_descriptor_with_no_target(self):
        """With no role/name the descriptor should indicate unknown."""
        step = _make_step(target_role="", target_name="")
        assert step.target_descriptor == "(unknown target)"


class TestTaskStepSerde:
    """Round-trip serialisation via to_dict / from_dict."""

    def test_round_trip(self):
        """to_dict → from_dict must reproduce an identical step."""
        original = _make_step(
            step_id="rt",
            action_type="type",
            target_role="textfield",
            target_name="Email",
            input_value="x@y.com",
            preconditions=["visible"],
            postconditions=["filled"],
            optional=False,
            timeout=5.0,
            depends_on=["prev"],
            metadata={"tag": "email"},
        )
        rebuilt = TaskStep.from_dict(original.to_dict())
        assert rebuilt.step_id == original.step_id
        assert rebuilt.action_type == original.action_type
        assert rebuilt.target_role == original.target_role
        assert rebuilt.target_name == original.target_name
        assert rebuilt.input_value == original.input_value
        assert rebuilt.preconditions == original.preconditions
        assert rebuilt.postconditions == original.postconditions
        assert rebuilt.optional == original.optional
        assert rebuilt.timeout == original.timeout
        assert rebuilt.depends_on == original.depends_on
        assert rebuilt.metadata == original.metadata

    def test_to_dict_omits_none_optionals(self):
        """Fields that are None or empty should be omitted from the dict."""
        step = _make_step()
        d = step.to_dict()
        assert "input_value" not in d
        assert "target_selector" not in d
        assert "timeout" not in d


# ===================================================================
# TaskFlow tests
# ===================================================================


class TestTaskFlow:
    """Tests for TaskFlow query helpers and serialisation."""

    def test_step_ids(self):
        """step_ids() should return IDs in insertion order."""
        s1, s2 = _make_step("a"), _make_step("b")
        flow = _make_flow(steps=[s1, s2])
        assert flow.step_ids() == ["a", "b"]

    def test_get_step_found(self):
        """get_step should return the step when the ID exists."""
        s = _make_step("target")
        flow = _make_flow(steps=[s])
        assert flow.get_step("target") is s

    def test_get_step_missing(self):
        """get_step should return None for unknown IDs."""
        flow = _make_flow(steps=[_make_step("x")])
        assert flow.get_step("nonexistent") is None

    def test_required_steps(self):
        """required_steps filters out optional steps."""
        r = _make_step("r", optional=False)
        o = _make_step("o", optional=True)
        flow = _make_flow(steps=[r, o])
        assert flow.required_steps() == [r]

    def test_input_steps(self):
        """input_steps returns only type/select actions."""
        t = _make_step("t", action_type="type", input_value="x")
        c = _make_step("c", action_type="click")
        flow = _make_flow(steps=[t, c])
        assert flow.input_steps() == [t]

    def test_action_type_counts(self):
        """action_type_counts returns a dict mapping action→count."""
        steps = [
            _make_step("a1", action_type="click"),
            _make_step("a2", action_type="click"),
            _make_step("a3", action_type="type", input_value="v"),
        ]
        flow = _make_flow(steps=steps)
        counts = flow.action_type_counts()
        assert counts["click"] == 2
        assert counts["type"] == 1

    def test_flow_round_trip(self):
        """to_dict → from_dict must preserve flow data."""
        flow = _make_flow(
            flow_id="f_rt",
            name="serde_flow",
            steps=[_make_step("s1"), _make_step("s2")],
            success_criteria=["done"],
            max_time=60.0,
            description="test",
            metadata={"k": "v"},
        )
        rebuilt = TaskFlow.from_dict(flow.to_dict())
        assert rebuilt.flow_id == flow.flow_id
        assert rebuilt.name == flow.name
        assert len(rebuilt.steps) == 2
        assert rebuilt.success_criteria == ["done"]
        assert rebuilt.max_time == 60.0

    def test_auto_generated_flow_id(self):
        """Omitting flow_id should produce a non-empty auto-generated ID."""
        flow = TaskFlow(name="anon")
        assert flow.flow_id.startswith("flow-")


# ===================================================================
# TaskSpec tests
# ===================================================================


class TestTaskSpec:
    """Tests for TaskSpec queries, validation, and serialisation."""

    def test_get_flow_found(self):
        """get_flow should return matching flow by ID."""
        spec = _login_spec()
        flow = spec.get_flow("login_flow")
        assert flow is not None
        assert flow.name == "standard_login"

    def test_get_flow_missing(self):
        """get_flow returns None when the ID is absent."""
        spec = _login_spec()
        assert spec.get_flow("nope") is None

    def test_total_steps(self):
        """total_steps counts steps across all flows."""
        spec = _login_spec()
        assert spec.total_steps() == 6  # 5 in main + 1 in alt

    def test_all_steps_iterator(self):
        """all_steps should yield every step from every flow."""
        spec = _login_spec()
        ids = [s.step_id for s in spec.all_steps()]
        assert "click_user" in ids
        assert "remember" in ids

    def test_all_target_names(self):
        """all_target_names collects unique names."""
        spec = _login_spec()
        names = spec.all_target_names()
        assert "Username" in names
        assert "Password" in names
        assert "Sign In" in names

    def test_all_target_roles(self):
        """all_target_roles collects unique roles."""
        spec = _login_spec()
        roles = spec.all_target_roles()
        assert "textfield" in roles
        assert "button" in roles
        assert "checkbox" in roles

    def test_validate_valid_spec(self):
        """A well-formed spec should produce no validation errors."""
        assert _login_spec().validate() == []

    def test_validate_empty_name_and_no_flows(self):
        """An empty name and no flows should both be reported."""
        errors = TaskSpec(name="", flows=[]).validate()
        assert any("name" in e.lower() for e in errors)
        assert any("no flows" in e.lower() for e in errors)

    def test_validate_missing_input_value(self):
        """A non-optional type step with no input_value should be flagged."""
        bad = _make_step("bad_type", action_type="type", input_value=None)
        errors = TaskSpec(name="bad", flows=[_make_flow(steps=[bad])]).validate()
        assert any("input" in e.lower() for e in errors)

    def test_validate_missing_dependency(self):
        """A step referencing a non-existent dependency should be flagged."""
        step = _make_step("orphan", depends_on=["ghost"])
        errors = TaskSpec(name="dep_err", flows=[_make_flow(steps=[step])]).validate()
        assert any("ghost" in e for e in errors)

    def test_to_dict_from_dict_round_trip(self):
        """Serialisation round-trip must reproduce an equivalent spec."""
        spec = _login_spec()
        rebuilt = TaskSpec.from_dict(spec.to_dict())
        assert rebuilt.spec_id == spec.spec_id
        assert rebuilt.name == spec.name
        assert rebuilt.description == spec.description
        assert len(rebuilt.flows) == len(spec.flows)
        assert rebuilt.initial_state == spec.initial_state
        assert rebuilt.metadata == spec.metadata

    def test_deep_copy_independence(self):
        """deep_copy() should return a structurally identical but
        independent object—mutations must not propagate."""
        spec = _login_spec()
        clone = spec.deep_copy()
        clone.name = "mutated"
        assert spec.name == "login_form"


# ===================================================================
# TaskGraph tests
# ===================================================================


class TestTaskGraph:
    """Tests for the dependency-DAG utilities in TaskGraph."""

    def _linear_flow(self) -> TaskFlow:
        """Three steps with implicit sequential dependencies."""
        return _make_flow(
            "linear",
            "linear",
            [_make_step("a"), _make_step("b"), _make_step("c")],
        )

    def _diamond_flow(self) -> TaskFlow:
        """Diamond DAG:  root → {mid1, mid2} → leaf."""
        root = _make_step("root")
        mid1 = _make_step("mid1", depends_on=["root"])
        mid2 = _make_step("mid2", depends_on=["root"])
        leaf = _make_step("leaf", depends_on=["mid1", "mid2"])
        return _make_flow("diamond", "diamond", [root, mid1, mid2, leaf])

    def test_from_flow_linear(self):
        """from_flow should produce a graph with all nodes."""
        g = TaskGraph.from_flow(self._linear_flow())
        assert set(g.nodes) == {"a", "b", "c"}

    def test_from_spec(self):
        """from_spec merges all flows into a single graph."""
        spec = _login_spec()
        g = TaskGraph.from_spec(spec)
        assert "click_user" in g.nodes
        assert "remember" in g.nodes

    def test_roots_and_leaves_linear(self):
        """In a linear chain the first step is the only root and the
        last step is the only leaf."""
        g = TaskGraph.from_flow(self._linear_flow())
        assert g.roots() == ["a"]
        assert g.leaves() == ["c"]

    def test_roots_and_leaves_diamond(self):
        """Diamond has a single root and a single leaf."""
        g = TaskGraph.from_flow(self._diamond_flow())
        assert g.roots() == ["root"]
        assert g.leaves() == ["leaf"]

    def test_topological_sort_linear(self):
        """Topological sort of a linear chain must preserve order."""
        g = TaskGraph.from_flow(self._linear_flow())
        order = g.topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_topological_sort_diamond(self):
        """In the diamond, root comes first and leaf comes last."""
        g = TaskGraph.from_flow(self._diamond_flow())
        order = g.topological_sort()
        assert order[0] == "root"
        assert order[-1] == "leaf"

    def test_critical_path_linear(self):
        """Critical path of a linear chain is the entire chain."""
        g = TaskGraph.from_flow(self._linear_flow())
        path = g.critical_path()
        assert path == ["a", "b", "c"]

    def test_critical_path_diamond(self):
        """Diamond critical path goes through root, one mid, and leaf."""
        g = TaskGraph.from_flow(self._diamond_flow())
        path = g.critical_path()
        assert path[0] == "root"
        assert path[-1] == "leaf"
        assert len(path) == 3

    def test_parallel_groups_diamond(self):
        """In the diamond, mid1 and mid2 should form a parallel group."""
        g = TaskGraph.from_flow(self._diamond_flow())
        groups = g.parallel_groups()
        mid_group = [grp for grp in groups if "mid1" in grp or "mid2" in grp]
        assert len(mid_group) == 1
        assert mid_group[0] == {"mid1", "mid2"}

    def test_parallel_groups_linear(self):
        """Linear chain has one step per group—no parallelism."""
        g = TaskGraph.from_flow(self._linear_flow())
        groups = g.parallel_groups()
        assert all(len(grp) == 1 for grp in groups)

    def test_empty_graph_topological_sort(self):
        """An empty graph should produce an empty topological order."""
        g = TaskGraph()
        assert g.topological_sort() == []

    def test_critical_path_with_weights(self):
        """Custom weights should shift the critical path selection."""
        g = TaskGraph.from_flow(self._diamond_flow())
        weights = {"root": 1.0, "mid1": 10.0, "mid2": 1.0, "leaf": 1.0}
        path = g.critical_path(weights=weights)
        assert "mid1" in path
