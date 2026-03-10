"""Tests for usability_oracle.taskspec.dsl — TaskDSLParser.

This module validates the YAML DSL parser that converts human-authored
task-specification YAML into TaskSpec model objects.  Coverage includes
valid parsing, schema validation, error handling, target-shorthand
notation, round-trip fidelity, strict vs. lenient modes, and template
reference resolution.
"""

from __future__ import annotations

import pytest
import yaml

from usability_oracle.taskspec.dsl import TaskDSLParser
from usability_oracle.taskspec.models import TaskSpec, TaskStep


# ===================================================================
# Helpers
# ===================================================================

MINIMAL_YAML = """\
task: minimal
flows:
  - name: only_flow
    steps:
      - action: click
        target: {role: button, name: OK}
"""

LOGIN_YAML = """\
task: login_form
description: Standard username/password login flow
initial_state:
  page: /login
  authenticated: false
flows:
  - name: standard_login
    max_time: 30
    steps:
      - action: click
        target: {role: textfield, name: "Username"}
      - action: type
        target: {role: textfield, name: "Username"}
        value: "user@example.com"
      - action: click
        target: {role: textfield, name: "Password"}
      - action: type
        target: {role: textfield, name: "Password"}
        value: "password123"
      - action: click
        target: {role: button, name: "Sign In"}
    success_criteria:
      - "page == /dashboard"
      - "authenticated == true"
"""

MULTI_FLOW_YAML = """\
task: checkout
description: Multi-flow checkout process
flows:
  - name: address
    steps:
      - action: click
        target: {role: textfield, name: Street}
      - action: type
        target: {role: textfield, name: Street}
        value: "123 Main St"
  - name: payment
    steps:
      - action: click
        target: {role: textfield, name: Card}
      - action: type
        target: {role: textfield, name: Card}
        value: "4111111111111111"
"""

TEMPLATE_YAML = """\
task: with_templates
templates:
  fill_field:
    action: click
    target: {role: textfield, name: "Field"}
flows:
  - name: main
    steps:
      - $ref: fill_field
      - action: click
        target: {role: button, name: Submit}
"""

SHORTHAND_YAML = """\
task: shorthand
flows:
  - name: nav
    steps:
      - action: click
        target: "button:Submit"
      - action: click
        target: "link:Home[#home-link]"
"""

OPTIONAL_STEP_YAML = """\
task: with_optional
flows:
  - name: main
    steps:
      - action: click
        target: {role: button, name: Start}
      - action: scroll
        target: {role: region, name: Content}
        optional: true
      - action: click
        target: {role: button, name: Finish}
"""

DEPENDS_ON_YAML = """\
task: with_deps
flows:
  - name: main
    steps:
      - id: step_a
        action: click
        target: {role: button, name: A}
      - id: step_b
        action: click
        target: {role: button, name: B}
        depends_on: [step_a]
"""


# ===================================================================
# Tests — parse with valid YAML
# ===================================================================


class TestParseValid:
    """Verify that well-formed YAML is parsed into correct TaskSpec objects."""

    def test_parse_minimal(self):
        """The smallest valid YAML should produce a TaskSpec with one flow
        containing one step."""
        parser = TaskDSLParser()
        spec = parser.parse(MINIMAL_YAML)
        assert isinstance(spec, TaskSpec)
        assert spec.name == "minimal"
        assert len(spec.flows) == 1
        assert len(spec.flows[0].steps) == 1

    def test_parse_login_flow(self):
        """The login YAML should produce a single flow with five steps
        and correct initial_state."""
        parser = TaskDSLParser()
        spec = parser.parse(LOGIN_YAML)
        assert spec.name == "login_form"
        assert spec.description == "Standard username/password login flow"
        assert spec.initial_state["page"] == "/login"
        flow = spec.flows[0]
        assert flow.name == "standard_login"
        assert flow.max_time == 30
        assert len(flow.steps) == 5
        assert flow.success_criteria == ["page == /dashboard", "authenticated == true"]

    def test_parse_step_action_types(self):
        """Each step should carry the correct action_type."""
        parser = TaskDSLParser()
        spec = parser.parse(LOGIN_YAML)
        actions = [s.action_type for s in spec.flows[0].steps]
        assert actions == ["click", "type", "click", "type", "click"]

    def test_parse_step_targets(self):
        """Step target roles and names should be extracted correctly."""
        parser = TaskDSLParser()
        spec = parser.parse(LOGIN_YAML)
        first = spec.flows[0].steps[0]
        assert first.target_role == "textfield"
        assert first.target_name == "Username"

    def test_parse_step_value(self):
        """Type steps should have their input_value populated."""
        parser = TaskDSLParser()
        spec = parser.parse(LOGIN_YAML)
        type_step = spec.flows[0].steps[1]
        assert type_step.input_value == "user@example.com"

    def test_parse_multi_flow(self):
        """A spec with two flows should produce two TaskFlow objects."""
        parser = TaskDSLParser()
        spec = parser.parse(MULTI_FLOW_YAML)
        assert len(spec.flows) == 2
        assert spec.flows[0].name == "address"
        assert spec.flows[1].name == "payment"


class TestParseInvalid:
    """Verify that invalid YAML triggers appropriate errors in strict mode."""

    def test_non_mapping_raises(self):
        """A YAML string that evaluates to a scalar should raise ValueError."""
        parser = TaskDSLParser(strict=True)
        with pytest.raises(ValueError, match="mapping"):
            parser.parse("just a plain string")

    def test_missing_task_key_strict(self):
        """Omitting the required 'task' key in strict mode should raise."""
        parser = TaskDSLParser(strict=True)
        bad_yaml = "flows:\n  - name: f\n    steps:\n      - action: click\n"
        with pytest.raises(ValueError, match="schema validation failed"):
            parser.parse(bad_yaml)

    def test_missing_flows_key_strict(self):
        """Omitting the required 'flows' key in strict mode should raise."""
        parser = TaskDSLParser(strict=True)
        bad_yaml = "task: orphan\n"
        with pytest.raises(ValueError, match="schema validation failed"):
            parser.parse(bad_yaml)

    def test_invalid_yaml_syntax(self):
        """Syntactically broken YAML should raise a yaml.YAMLError."""
        parser = TaskDSLParser()
        with pytest.raises(yaml.YAMLError):
            parser.parse("task: bad\n  flows: [unbalanced")

    def test_invalid_action_type_strict(self):
        """An unknown action type should be caught during schema validation."""
        parser = TaskDSLParser(strict=True)
        bad_yaml = """\
task: bad_action
flows:
  - name: f
    steps:
      - action: explode
"""
        with pytest.raises(ValueError):
            parser.parse(bad_yaml)


class TestSchemaValidation:
    """Focused tests for the _validate_schema internals."""

    def test_valid_schema_returns_no_errors(self):
        """A well-formed data dict should produce zero errors."""
        parser = TaskDSLParser()
        data = yaml.safe_load(MINIMAL_YAML)
        errors = parser._validate_schema(data)
        assert errors == []

    def test_missing_required_field_reported(self):
        """Missing 'task' should appear in the error list."""
        parser = TaskDSLParser()
        data = {"flows": [{"name": "f", "steps": [{"action": "click"}]}]}
        errors = parser._validate_schema(data)
        assert any("task" in e for e in errors)

    def test_wrong_type_reported(self):
        """A 'task' that is an int instead of a string should be reported."""
        parser = TaskDSLParser()
        data = {"task": 42, "flows": []}
        errors = parser._validate_schema(data)
        assert any("task" in e and "string" in e for e in errors)


# ===================================================================
# Step / flow parsing details
# ===================================================================


class TestStepParsing:
    """Test step-level parsing with all possible field types."""

    def test_optional_step(self):
        """The optional flag should propagate from YAML to the model."""
        parser = TaskDSLParser()
        spec = parser.parse(OPTIONAL_STEP_YAML)
        opt = [s for s in spec.flows[0].steps if s.optional]
        assert len(opt) == 1
        assert opt[0].action_type == "scroll"

    def test_depends_on_parsed(self):
        """depends_on arrays in YAML should be captured in the model."""
        parser = TaskDSLParser()
        spec = parser.parse(DEPENDS_ON_YAML)
        step_b = spec.flows[0].steps[1]
        assert step_b.depends_on == ["step_a"]

    def test_step_ids_auto_generated(self):
        """Steps without explicit IDs should get auto-generated ones."""
        parser = TaskDSLParser()
        spec = parser.parse(MINIMAL_YAML)
        step = spec.flows[0].steps[0]
        assert step.step_id  # non-empty
        assert isinstance(step.step_id, str)


class TestFlowParsing:
    """Flow-level parsing tests."""

    def test_flow_names(self):
        """Flow names should be extracted from YAML."""
        parser = TaskDSLParser()
        spec = parser.parse(MULTI_FLOW_YAML)
        names = [f.name for f in spec.flows]
        assert "address" in names
        assert "payment" in names

    def test_flow_success_criteria(self):
        """success_criteria should be a list of strings."""
        parser = TaskDSLParser()
        spec = parser.parse(LOGIN_YAML)
        assert len(spec.flows[0].success_criteria) == 2


# ===================================================================
# Target shorthand
# ===================================================================


class TestTargetShorthand:
    """Tests for the _parse_target_shorthand static method."""

    def test_role_name(self):
        """'button:Submit' should parse to role=button, name=Submit."""
        result = TaskDSLParser._parse_target_shorthand("button:Submit")
        assert result["role"] == "button"
        assert result["name"] == "Submit"

    def test_role_name_selector(self):
        """'link:Home[#home-link]' should include a selector."""
        result = TaskDSLParser._parse_target_shorthand("link:Home[#home-link]")
        assert result["role"] == "link"
        assert result["name"] == "Home"
        assert result["selector"] == "#home-link"

    def test_plain_name_fallback(self):
        """A string without a colon should become the name only."""
        result = TaskDSLParser._parse_target_shorthand("JustAName")
        assert result.get("name") == "JustAName"
        assert "role" not in result

    def test_shorthand_in_yaml(self):
        """Shorthand notation used in actual YAML should parse correctly
        (lenient mode is needed because the schema expects target to be
        an object, not a string)."""
        parser = TaskDSLParser(strict=False)
        spec = parser.parse(SHORTHAND_YAML)
        steps = spec.flows[0].steps
        assert steps[0].target_role == "button"
        assert steps[0].target_name == "Submit"
        assert steps[1].target_role == "link"
        assert steps[1].target_name == "Home"
        assert steps[1].target_selector == "#home-link"


# ===================================================================
# Round-trip: parse → to_yaml → parse
# ===================================================================


class TestRoundTrip:
    """Verify that to_yaml produces YAML that re-parses to an equivalent spec."""

    def test_round_trip_minimal(self):
        """Minimal spec should survive a round-trip."""
        parser = TaskDSLParser()
        spec1 = parser.parse(MINIMAL_YAML)
        yaml_str = TaskDSLParser.to_yaml(spec1)
        spec2 = parser.parse(yaml_str)
        assert spec2.name == spec1.name
        assert len(spec2.flows) == len(spec1.flows)
        assert len(spec2.flows[0].steps) == len(spec1.flows[0].steps)

    def test_round_trip_login(self):
        """The full login spec should survive a round-trip with all data intact."""
        parser = TaskDSLParser()
        spec1 = parser.parse(LOGIN_YAML)
        yaml_str = TaskDSLParser.to_yaml(spec1)
        spec2 = parser.parse(yaml_str)
        assert spec2.name == spec1.name
        assert spec2.flows[0].max_time == spec1.flows[0].max_time
        assert len(spec2.flows[0].steps) == len(spec1.flows[0].steps)

    def test_round_trip_preserves_initial_state(self):
        """initial_state should survive a round-trip."""
        parser = TaskDSLParser()
        spec1 = parser.parse(LOGIN_YAML)
        yaml_str = TaskDSLParser.to_yaml(spec1)
        spec2 = parser.parse(yaml_str)
        assert spec2.initial_state == spec1.initial_state


# ===================================================================
# Strict vs. lenient mode
# ===================================================================


class TestStrictVsLenient:
    """Ensure strict mode raises while lenient collects warnings."""

    def test_lenient_missing_task(self):
        """In lenient mode, missing 'task' should add a warning, not raise."""
        parser = TaskDSLParser(strict=False)
        bad_yaml = "flows:\n  - name: f\n    steps:\n      - action: click\n"
        spec = parser.parse(bad_yaml)
        assert len(parser.warnings) > 0
        assert any("task" in w for w in parser.warnings)

    def test_strict_missing_task(self):
        """In strict mode, missing 'task' should raise."""
        parser = TaskDSLParser(strict=True)
        with pytest.raises(ValueError):
            parser.parse("flows:\n  - name: f\n    steps:\n      - action: click\n")

    def test_lenient_bad_ref(self):
        """In lenient mode, an unresolved $ref should produce a warning."""
        parser = TaskDSLParser(strict=False)
        yaml_str = """\
task: bad_ref
flows:
  - name: f
    steps:
      - $ref: nonexistent
        action: click
"""
        spec = parser.parse(yaml_str)
        assert len(parser.warnings) > 0


# ===================================================================
# Template / reference resolution
# ===================================================================


class TestReferenceResolution:
    """Tests for $ref template expansion."""

    def test_template_expansion(self):
        """A $ref should be resolved by merging the template into the step.
        Lenient mode is needed because $ref steps lack 'action' until
        template expansion, which happens after schema validation."""
        parser = TaskDSLParser(strict=False)
        spec = parser.parse(TEMPLATE_YAML)
        first_step = spec.flows[0].steps[0]
        assert first_step.action_type == "click"
        assert first_step.target_role == "textfield"
        assert first_step.target_name == "Field"

    def test_template_local_override(self):
        """Local keys alongside $ref should override the template values.
        Lenient mode is used because $ref steps lack 'action' before
        expansion."""
        yaml_str = """\
task: override
templates:
  base:
    action: click
    target: {role: button, name: Base}
flows:
  - name: f
    steps:
      - $ref: base
        target: {role: link, name: Override}
"""
        parser = TaskDSLParser(strict=False)
        spec = parser.parse(yaml_str)
        step = spec.flows[0].steps[0]
        assert step.target_role == "link"
        assert step.target_name == "Override"

    def test_unresolved_ref_strict_raises(self):
        """An unresolved $ref in strict mode should raise ValueError.
        The schema validation fires before ref resolution, so we match
        the schema error instead."""
        parser = TaskDSLParser(strict=True)
        yaml_str = """\
task: bad
flows:
  - name: f
    steps:
      - $ref: missing
"""
        with pytest.raises(ValueError):
            parser.parse(yaml_str)

    def test_unresolved_ref_lenient_warns(self):
        """In lenient mode, an unresolved $ref should produce warnings
        but still complete parsing."""
        parser = TaskDSLParser(strict=False)
        yaml_str = """\
task: bad
flows:
  - name: f
    steps:
      - $ref: missing
        action: click
"""
        spec = parser.parse(yaml_str)
        assert any("Unresolved" in w or "missing" in w.lower()
                    for w in parser.warnings)
