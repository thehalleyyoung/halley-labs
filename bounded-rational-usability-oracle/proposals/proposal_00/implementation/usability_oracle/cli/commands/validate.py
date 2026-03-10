"""
usability_oracle.cli.commands.validate — Validate task specification files.

Implements the ``usability-oracle validate`` command which checks task
specifications for syntactic correctness and optionally verifies them
against a UI source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)


def validate_command(
    task_spec_file: str,
    ui_source: Optional[str] = None,
) -> int:
    """Validate a task specification file.

    Parameters
    ----------
    task_spec_file : str
        Path to the task specification (YAML or JSON).
    ui_source : str, optional
        Path to a UI source to validate the spec against.

    Returns
    -------
    int
        Exit code: 0 = valid, 1 = validation errors, 2 = error.
    """
    try:
        spec_path = Path(task_spec_file)
        if not spec_path.exists():
            click.echo(f"Error: File not found: {spec_path}", err=True)
            return 2

        # Load task spec
        spec_text = spec_path.read_text(encoding="utf-8")
        if spec_path.suffix in (".yaml", ".yml"):
            import yaml
            spec_data = yaml.safe_load(spec_text)
        else:
            spec_data = json.loads(spec_text)

        click.echo(f"Validating {spec_path.name}…\n")

        errors: list[str] = []
        warnings: list[str] = []

        # Structural validation
        _validate_structure(spec_data, errors, warnings)

        # Flow validation
        flows = spec_data.get("flows", [])
        if not flows:
            single_flow = spec_data.get("steps")
            if single_flow:
                flows = [{"steps": single_flow, "name": "default"}]
            else:
                errors.append("No flows or steps defined in task spec")

        for i, flow in enumerate(flows):
            flow_name = flow.get("name", f"flow_{i}")
            _validate_flow(flow, flow_name, errors, warnings)

        # Cross-reference with UI source if provided
        if ui_source:
            _validate_against_ui(spec_data, ui_source, errors, warnings)

        # Report results
        if errors:
            click.echo(click.style("Errors:", fg="red", bold=True))
            for err in errors:
                click.echo(f"  ✗ {err}")
            click.echo()

        if warnings:
            click.echo(click.style("Warnings:", fg="yellow"))
            for warn in warnings:
                click.echo(f"  ⚠ {warn}")
            click.echo()

        if not errors and not warnings:
            click.echo(click.style("✓ Task specification is valid", fg="green", bold=True))
            return 0
        elif errors:
            click.echo(
                click.style(
                    f"✗ {len(errors)} error(s), {len(warnings)} warning(s)",
                    fg="red",
                )
            )
            return 1
        else:
            click.echo(
                click.style(
                    f"✓ Valid with {len(warnings)} warning(s)",
                    fg="yellow",
                )
            )
            return 0

    except json.JSONDecodeError as exc:
        click.echo(f"Error: Invalid JSON: {exc}", err=True)
        return 2
    except Exception as exc:
        logger.exception("Validate command failed")
        click.echo(f"Error: {exc}", err=True)
        return 2


def _validate_structure(
    spec: Any, errors: list[str], warnings: list[str]
) -> None:
    """Validate top-level task spec structure."""
    if not isinstance(spec, dict):
        errors.append(f"Task spec must be a dict, got {type(spec).__name__}")
        return

    # Required fields
    if "name" not in spec and "flows" not in spec and "steps" not in spec:
        errors.append("Task spec must have 'name', 'flows', or 'steps'")

    # Optional metadata
    if "description" not in spec:
        warnings.append("Consider adding a 'description' field")

    if "version" not in spec:
        warnings.append("Consider adding a 'version' field")


def _validate_flow(
    flow: dict[str, Any],
    flow_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single task flow."""
    steps = flow.get("steps", [])
    if not steps:
        errors.append(f"Flow '{flow_name}' has no steps")
        return

    seen_ids: set[str] = set()
    valid_action_types = {
        "click", "type", "select", "scroll", "navigate",
        "read", "wait", "verify", "hover", "drag",
    }

    for i, step in enumerate(steps):
        step_id = step.get("step_id", step.get("id", f"step_{i}"))

        # Duplicate ID check
        if step_id in seen_ids:
            errors.append(
                f"Flow '{flow_name}': Duplicate step ID '{step_id}'"
            )
        seen_ids.add(step_id)

        # Action type validation
        action_type = step.get("action_type", step.get("action"))
        if not action_type:
            errors.append(
                f"Flow '{flow_name}', step '{step_id}': "
                f"Missing action_type"
            )
        elif action_type not in valid_action_types:
            warnings.append(
                f"Flow '{flow_name}', step '{step_id}': "
                f"Unknown action_type '{action_type}'"
            )

        # Target validation
        target_types = {"click", "type", "select", "read", "hover", "drag"}
        if action_type in target_types:
            has_target = (
                step.get("target_selector")
                or step.get("target_name")
                or step.get("target_role")
            )
            if not has_target:
                errors.append(
                    f"Flow '{flow_name}', step '{step_id}': "
                    f"Action '{action_type}' requires a target "
                    f"(target_selector, target_name, or target_role)"
                )

        # Input value for type actions
        if action_type == "type" and not step.get("input_value"):
            warnings.append(
                f"Flow '{flow_name}', step '{step_id}': "
                f"Type action without input_value"
            )

        # Dependency validation
        depends_on = step.get("depends_on", [])
        for dep in depends_on:
            if dep not in seen_ids:
                warnings.append(
                    f"Flow '{flow_name}', step '{step_id}': "
                    f"Dependency '{dep}' not yet defined "
                    f"(forward reference or missing step)"
                )


def _validate_against_ui(
    spec: dict[str, Any],
    ui_source_path: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate task spec targets against a UI accessibility tree."""
    try:
        from usability_oracle.accessibility import (
            HTMLAccessibilityParser,
            JSONAccessibilityParser,
        )

        source_path = Path(ui_source_path)
        source_content = source_path.read_text(encoding="utf-8")

        if source_content.strip().startswith("{"):
            parser = JSONAccessibilityParser()
        else:
            parser = HTMLAccessibilityParser()

        tree = parser.parse(source_content)

        # Check that referenced targets exist in the tree
        flows = spec.get("flows", [])
        if not flows and spec.get("steps"):
            flows = [{"steps": spec["steps"]}]

        node_names = {n.name.lower() for n in tree.node_index.values() if n.name}
        node_roles = {n.role for n in tree.node_index.values()}
        node_ids = set(tree.node_index.keys())

        for flow in flows:
            for step in flow.get("steps", []):
                target_name = step.get("target_name", "")
                target_selector = step.get("target_selector", "")
                target_role = step.get("target_role", "")

                if target_selector and target_selector not in node_ids:
                    warnings.append(
                        f"Target selector '{target_selector}' not found in UI"
                    )
                if target_name and target_name.lower() not in node_names:
                    warnings.append(
                        f"Target name '{target_name}' not found in UI"
                    )
                if target_role and target_role not in node_roles:
                    warnings.append(
                        f"Target role '{target_role}' not found in UI"
                    )

    except Exception as exc:
        warnings.append(f"Could not validate against UI: {exc}")
