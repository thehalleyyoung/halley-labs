"""
usability_oracle.taskspec.dsl — YAML DSL parser for task specifications.

Defines a clean, human-authored YAML schema for specifying UI task flows
and converts them into :class:`TaskSpec` model objects.

Example YAML
-------------

.. code-block:: yaml

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
            value: "user@example.com"
          - action: click
            target: {role: textfield, name: "Password"}
          - action: type
            value: "password123"
          - action: click
            target: {role: button, name: "Sign In"}
        success_criteria:
          - "page == /dashboard"
          - "authenticated == true"

The parser handles:
* ``$ref`` / YAML anchors for reusable step snippets
* Schema validation
* Auto-generated IDs where omitted
"""

from __future__ import annotations

import copy
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from usability_oracle.taskspec.models import (
    ACTION_TYPES,
    TaskFlow,
    TaskSpec,
    TaskStep,
)

# ---------------------------------------------------------------------------
# YAML schema definition (for validation)
# ---------------------------------------------------------------------------

_STEP_SCHEMA: Dict[str, Any] = {
    "required": ["action"],
    "properties": {
        "id": {"type": "string"},
        "action": {"type": "string", "enum": sorted(ACTION_TYPES)},
        "target": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "name": {"type": "string"},
                "selector": {"type": "string"},
            },
        },
        "value": {"type": "string"},
        "preconditions": {"type": "array", "items": {"type": "string"}},
        "postconditions": {"type": "array", "items": {"type": "string"}},
        "optional": {"type": "boolean"},
        "description": {"type": "string"},
        "timeout": {"type": "number"},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
}

_FLOW_SCHEMA: Dict[str, Any] = {
    "required": ["name", "steps"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "steps": {"type": "array", "items": _STEP_SCHEMA},
        "success_criteria": {"type": "array", "items": {"type": "string"}},
        "max_time": {"type": "number"},
        "description": {"type": "string"},
        "metadata": {"type": "object"},
    },
}

_SPEC_SCHEMA: Dict[str, Any] = {
    "required": ["task", "flows"],
    "properties": {
        "task": {"type": "string"},
        "description": {"type": "string"},
        "flows": {"type": "array", "items": _FLOW_SCHEMA},
        "initial_state": {"type": "object"},
        "metadata": {"type": "object"},
        "templates": {"type": "object"},
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_type(value: Any, expected: str) -> bool:
    """Lightweight type check for schema validation."""
    type_map = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return isinstance(value, type_map.get(expected, object))


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# TaskDSLParser
# ---------------------------------------------------------------------------


class TaskDSLParser:
    """Parse YAML task specifications into :class:`TaskSpec` objects.

    Usage::

        parser = TaskDSLParser()
        spec = parser.parse(yaml_string)
        # or
        spec = parser.parse_file(Path("tasks/login.yaml"))
    """

    SCHEMA: Dict[str, Any] = _SPEC_SCHEMA

    def __init__(self, *, strict: bool = True) -> None:
        """
        Parameters
        ----------
        strict : bool
            If *True*, raise on schema validation errors.  If *False*,
            collect warnings and continue with best-effort parsing.
        """
        self._strict = strict
        self._warnings: List[str] = []
        self._templates: Dict[str, Dict[str, Any]] = {}

    @property
    def warnings(self) -> List[str]:
        """Warnings accumulated during the last parse."""
        return list(self._warnings)

    # -- public API ----------------------------------------------------------

    def parse(self, yaml_str: str) -> TaskSpec:
        """Parse a YAML string into a :class:`TaskSpec`.

        Parameters
        ----------
        yaml_str : str
            Raw YAML text.

        Returns
        -------
        TaskSpec

        Raises
        ------
        ValueError
            If *strict* mode is enabled and the YAML violates the schema.
        yaml.YAMLError
            If the YAML syntax is invalid.
        """
        self._warnings = []
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML element must be a mapping.")

        errors = self._validate_schema(data)
        if errors and self._strict:
            raise ValueError(
                "YAML schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        elif errors:
            self._warnings.extend(errors)

        # Extract inline templates (reusable step snippets)
        self._templates = data.get("templates", {})

        spec = self._build_spec(data)
        spec = self._resolve_references(spec, data)
        return spec

    def parse_file(self, path: Path) -> TaskSpec:
        """Read a YAML file and parse it.

        Parameters
        ----------
        path : Path
            Path to the ``.yaml`` / ``.yml`` file.
        """
        text = Path(path).read_text(encoding="utf-8")
        return self.parse(text)

    # -- schema validation ---------------------------------------------------

    def _validate_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate *data* against :attr:`SCHEMA`.  Return error messages."""
        return self._validate_node(data, self.SCHEMA, path="")

    def _validate_node(
        self, data: Any, schema: Dict[str, Any], path: str
    ) -> List[str]:
        errors: List[str] = []
        if not isinstance(data, dict):
            errors.append(f"{path or 'root'}: expected mapping, got {type(data).__name__}")
            return errors

        # required fields
        for field_name in schema.get("required", []):
            if field_name not in data:
                errors.append(f"{path or 'root'}: missing required field '{field_name}'")

        # property types
        props = schema.get("properties", {})
        for key, value in data.items():
            if key.startswith("$"):
                continue  # skip $ref and similar
            if key not in props:
                continue  # unknown keys are allowed (open schema)
            prop_schema = props[key]
            expected_type = prop_schema.get("type")

            if expected_type and not _check_type(value, expected_type):
                errors.append(
                    f"{path}.{key}: expected {expected_type}, got {type(value).__name__}"
                )
                continue

            # enum check
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                errors.append(
                    f"{path}.{key}: value {value!r} not in {prop_schema['enum']}"
                )

            # recurse into arrays
            if expected_type == "array" and isinstance(value, list):
                item_schema = prop_schema.get("items", {})
                if item_schema:
                    for i, item in enumerate(value):
                        if isinstance(item_schema, dict) and "properties" in item_schema:
                            errors.extend(
                                self._validate_node(item, item_schema, f"{path}.{key}[{i}]")
                            )

            # recurse into objects
            if expected_type == "object" and isinstance(value, dict) and "properties" in prop_schema:
                errors.extend(
                    self._validate_node(value, prop_schema, f"{path}.{key}")
                )

        return errors

    # -- building model objects ----------------------------------------------

    def _build_spec(self, data: Dict[str, Any]) -> TaskSpec:
        flows = [self._parse_flow(f) for f in data.get("flows", [])]
        return TaskSpec(
            spec_id=data.get("id", _gen_id("spec")),
            name=data.get("task", ""),
            description=data.get("description", ""),
            flows=flows,
            initial_state=dict(data.get("initial_state", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def _parse_flow(self, data: Dict[str, Any]) -> TaskFlow:
        """Parse a single flow mapping into a :class:`TaskFlow`."""
        if not isinstance(data, dict):
            raise ValueError(f"Flow entry must be a mapping, got {type(data).__name__}")

        steps = [self._parse_step(s) for s in data.get("steps", [])]
        return TaskFlow(
            flow_id=data.get("id", _gen_id("flow")),
            name=data.get("name", ""),
            steps=steps,
            success_criteria=list(data.get("success_criteria", [])),
            max_time=data.get("max_time"),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
        )

    def _parse_step(self, data: Dict[str, Any]) -> TaskStep:
        """Parse a single step mapping into a :class:`TaskStep`."""
        if not isinstance(data, dict):
            raise ValueError(f"Step entry must be a mapping, got {type(data).__name__}")

        # Handle $ref (template reference)
        if "$ref" in data:
            data = self._expand_ref(data)

        target = data.get("target", {})
        if isinstance(target, str):
            # shorthand: "button:Submit" -> role=button, name=Submit
            target = self._parse_target_shorthand(target)

        return TaskStep(
            step_id=data.get("id", _gen_id("step")),
            action_type=data.get("action", "click"),
            target_role=target.get("role", ""),
            target_name=target.get("name", ""),
            target_selector=target.get("selector"),
            input_value=data.get("value"),
            preconditions=list(data.get("preconditions", [])),
            postconditions=list(data.get("postconditions", [])),
            optional=bool(data.get("optional", False)),
            description=data.get("description", ""),
            timeout=data.get("timeout"),
            depends_on=list(data.get("depends_on", [])),
            metadata=dict(data.get("metadata", {})),
        )

    # -- reference resolution ------------------------------------------------

    def _expand_ref(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a ``$ref`` key by looking up ``self._templates``."""
        ref = data["$ref"]
        if ref not in self._templates:
            if self._strict:
                raise ValueError(f"Unresolved $ref: {ref!r}")
            self._warnings.append(f"Unresolved $ref: {ref!r}")
            return data
        resolved = copy.deepcopy(self._templates[ref])
        # local overrides take precedence
        for k, v in data.items():
            if k != "$ref":
                resolved[k] = v
        return resolved

    def _resolve_references(self, spec: TaskSpec, raw_data: Dict[str, Any]) -> TaskSpec:
        """Post-processing pass to resolve any remaining references.

        Handles:
        - YAML anchors (already handled by pyyaml)
        - $ref expansions (handled during step parsing)
        - Cross-flow step references (step IDs prefixed with ``flow_id.``)
        """
        # Build a global step-ID lookup for cross-flow references
        all_steps: Dict[str, TaskStep] = {}
        for flow in spec.flows:
            for step in flow.steps:
                all_steps[step.step_id] = step
                # also register with flow-qualified name
                all_steps[f"{flow.flow_id}.{step.step_id}"] = step

        # Resolve cross-flow depends_on references
        for flow in spec.flows:
            for step in flow.steps:
                resolved_deps: List[str] = []
                for dep in step.depends_on:
                    if dep in all_steps:
                        resolved_deps.append(dep)
                    elif f"{flow.flow_id}.{dep}" in all_steps:
                        resolved_deps.append(dep)
                    else:
                        self._warnings.append(
                            f"Step {step.step_id!r}: dependency {dep!r} not found."
                        )
                        resolved_deps.append(dep)
                step.depends_on = resolved_deps

        return spec

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _parse_target_shorthand(target_str: str) -> Dict[str, str]:
        """Parse shorthand target notation ``role:name`` or ``role:name[selector]``."""
        result: Dict[str, str] = {}
        m = re.match(r"^(\w+):(.+?)(?:\[(.+)\])?$", target_str.strip())
        if m:
            result["role"] = m.group(1)
            result["name"] = m.group(2).strip()
            if m.group(3):
                result["selector"] = m.group(3).strip()
        else:
            result["name"] = target_str.strip()
        return result

    # -- multi-file support --------------------------------------------------

    def parse_directory(self, directory: Path) -> List[TaskSpec]:
        """Parse all ``*.yaml`` and ``*.yml`` files in *directory*."""
        specs: List[TaskSpec] = []
        dirpath = Path(directory)
        for path in sorted(dirpath.glob("*.y*ml")):
            if path.suffix in {".yaml", ".yml"}:
                specs.append(self.parse_file(path))
        return specs

    # -- YAML rendering (round-trip) -----------------------------------------

    @staticmethod
    def to_yaml(spec: TaskSpec) -> str:
        """Serialise a :class:`TaskSpec` back to YAML DSL format."""
        data: Dict[str, Any] = {
            "task": spec.name,
        }
        if spec.description:
            data["description"] = spec.description
        if spec.initial_state:
            data["initial_state"] = spec.initial_state
        if spec.metadata:
            data["metadata"] = spec.metadata

        flows: List[Dict[str, Any]] = []
        for flow in spec.flows:
            fdata: Dict[str, Any] = {"name": flow.name}
            if flow.max_time is not None:
                fdata["max_time"] = flow.max_time
            if flow.description:
                fdata["description"] = flow.description

            steps: List[Dict[str, Any]] = []
            for step in flow.steps:
                sdata: Dict[str, Any] = {"action": step.action_type}
                target: Dict[str, str] = {}
                if step.target_role:
                    target["role"] = step.target_role
                if step.target_name:
                    target["name"] = step.target_name
                if step.target_selector:
                    target["selector"] = step.target_selector
                if target:
                    sdata["target"] = target
                if step.input_value is not None:
                    sdata["value"] = step.input_value
                if step.optional:
                    sdata["optional"] = True
                if step.description:
                    sdata["description"] = step.description
                if step.preconditions:
                    sdata["preconditions"] = step.preconditions
                if step.postconditions:
                    sdata["postconditions"] = step.postconditions
                if step.timeout is not None:
                    sdata["timeout"] = step.timeout
                if step.depends_on:
                    sdata["depends_on"] = step.depends_on
                steps.append(sdata)

            fdata["steps"] = steps
            if flow.success_criteria:
                fdata["success_criteria"] = flow.success_criteria
            flows.append(fdata)

        data["flows"] = flows
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
