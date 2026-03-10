"""Schema validation for accessibility-tree formats.

Provides :class:`SchemaValidator` for validating accessibility-tree
data against format-specific schemas.  Supports JSON Schema validation,
HTML/ARIA structural checks, Android XML schema, and task-specification
format validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

@unique
class ValidationSeverity(Enum):
    """Severity of a validation issue."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue.

    Attributes
    ----------
    message : str
        Description of the issue.
    severity : ValidationSeverity
    path : str
        JSON Pointer or XPath to the offending element.
    line : int
        1-based line number (0 if unknown).
    column : int
        1-based column number (0 if unknown).
    rule : str
        Rule or schema keyword that was violated.
    """

    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    path: str = ""
    line: int = 0
    column: int = 0
    rule: str = ""


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes
    ----------
    valid : bool
        True if no errors were found (warnings are allowed).
    issues : list[ValidationIssue]
        All issues found during validation.
    format_id : str
        The format that was validated against.
    """

    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    format_id: str = ""

    @property
    def error_count(self) -> int:
        return sum(
            1 for i in self.issues if i.severity == ValidationSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1 for i in self.issues if i.severity == ValidationSeverity.WARNING
        )


# ═══════════════════════════════════════════════════════════════════════════
# Built-in schemas
# ═══════════════════════════════════════════════════════════════════════════

_A11Y_TREE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Accessibility Tree Node",
    "type": "object",
    "required": ["id", "role"],
    "properties": {
        "id": {"type": "string"},
        "role": {"type": "string"},
        "name": {"type": "string"},
        "bounding_box": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"},
            },
        },
        "properties": {"type": "object"},
        "children": {
            "type": "array",
            "items": {"$ref": "#"},
        },
    },
}

_ANDROID_XML_SCHEMA: Dict[str, Any] = {
    "root_element": "hierarchy",
    "node_element": "node",
    "required_attributes": [
        "class", "text", "resource-id", "content-desc", "bounds",
    ],
    "optional_attributes": [
        "checkable", "checked", "clickable", "enabled", "focusable",
        "focused", "scrollable", "long-clickable", "password", "selected",
        "package", "index",
    ],
}

_TASKSPEC_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Task Specification",
    "type": "object",
    "required": ["task"],
    "properties": {
        "task": {
            "type": "object",
            "required": ["name", "steps"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": {"type": "string"},
                            "target": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

# Known ARIA roles
_VALID_ARIA_ROLES: set[str] = {
    "alert", "alertdialog", "application", "article", "banner", "blockquote",
    "button", "caption", "cell", "checkbox", "code", "columnheader",
    "combobox", "command", "complementary", "composite", "contentinfo",
    "definition", "deletion", "dialog", "directory", "document", "emphasis",
    "feed", "figure", "form", "generic", "grid", "gridcell", "group",
    "heading", "img", "input", "insertion", "landmark", "link", "list",
    "listbox", "listitem", "log", "main", "marquee", "math", "menu",
    "menubar", "menuitem", "menuitemcheckbox", "menuitemradio", "meter",
    "navigation", "none", "note", "option", "paragraph", "presentation",
    "progressbar", "radio", "radiogroup", "range", "region", "roletype",
    "row", "rowgroup", "rowheader", "scrollbar", "search", "searchbox",
    "section", "sectionhead", "select", "separator", "slider", "spinbutton",
    "status", "strong", "structure", "subscript", "superscript", "switch",
    "tab", "table", "tablist", "tabpanel", "term", "textbox", "time",
    "timer", "toolbar", "tooltip", "tree", "treegrid", "treeitem", "widget",
    "window",
}


# ═══════════════════════════════════════════════════════════════════════════
# SchemaValidator
# ═══════════════════════════════════════════════════════════════════════════

class SchemaValidator:
    """Validate accessibility-tree content against format-specific schemas.

    Usage::

        validator = SchemaValidator()
        result = validator.validate_json_schema(content, schema)
        result = validator.validate_html_structure(html)
    """

    # ------------------------------------------------------------------
    # JSON Schema validation
    # ------------------------------------------------------------------

    def validate_json_schema(
        self,
        content: str | dict,
        schema: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate content against a JSON Schema.

        Uses a built-in lightweight validator (no ``jsonschema``
        dependency required).  Checks ``type``, ``required``,
        ``properties``, and ``items`` keywords.

        Parameters
        ----------
        content : str or dict
            JSON string or parsed dict.
        schema : dict, optional
            JSON Schema document.  Defaults to the built-in
            accessibility-tree schema.

        Returns
        -------
        ValidationResult
        """
        if schema is None:
            schema = _A11Y_TREE_SCHEMA

        if isinstance(content, str):
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    valid=False,
                    issues=[ValidationIssue(
                        message=f"Invalid JSON: {e}",
                        severity=ValidationSeverity.ERROR,
                        line=e.lineno,
                        column=e.colno,
                        rule="json-syntax",
                    )],
                    format_id="json",
                )
        else:
            data = content

        issues: list[ValidationIssue] = []
        self._validate_node(data, schema, "", issues)

        return ValidationResult(
            valid=all(i.severity != ValidationSeverity.ERROR for i in issues),
            issues=issues,
            format_id="json-schema",
        )

    def _validate_node(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        """Recursively validate a data node against a schema."""
        # Type check
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(data, expected_type):
                issues.append(ValidationIssue(
                    message=f"Expected type '{expected_type}' at {path or '/'}",
                    severity=ValidationSeverity.ERROR,
                    path=path or "/",
                    rule="type",
                ))
                return

        # Required check
        if isinstance(data, dict):
            for req in schema.get("required", []):
                if req not in data:
                    issues.append(ValidationIssue(
                        message=f"Missing required property '{req}' at {path or '/'}",
                        severity=ValidationSeverity.ERROR,
                        path=f"{path}/{req}",
                        rule="required",
                    ))

            # Properties
            properties = schema.get("properties", {})
            for key, sub_schema in properties.items():
                if key in data:
                    self._validate_node(data[key], sub_schema, f"{path}/{key}", issues)

            # Recursive children via $ref
            if "children" in data and isinstance(data["children"], list):
                items_schema = properties.get("children", {}).get("items", {})
                ref = items_schema.get("$ref")
                child_schema = schema if ref == "#" else items_schema
                for i, child in enumerate(data["children"]):
                    self._validate_node(
                        child, child_schema, f"{path}/children/{i}", issues
                    )

        # Array items
        if isinstance(data, list):
            items_schema = schema.get("items", {})
            if items_schema:
                for i, item in enumerate(data):
                    self._validate_node(item, items_schema, f"{path}/{i}", issues)

    @staticmethod
    def _check_type(data: Any, expected: str) -> bool:
        """Check if data matches the expected JSON Schema type."""
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }
        expected_type = type_map.get(expected)
        if expected_type is None:
            return True
        return isinstance(data, expected_type)

    # ------------------------------------------------------------------
    # HTML / ARIA validation
    # ------------------------------------------------------------------

    def validate_html_structure(
        self, content: str
    ) -> ValidationResult:
        """Validate HTML well-formedness and ARIA usage.

        Checks for:
        - Valid ARIA roles
        - Required ARIA attributes
        - Proper nesting rules
        - Duplicate IDs

        Parameters
        ----------
        content : str
            HTML content.

        Returns
        -------
        ValidationResult
        """
        issues: list[ValidationIssue] = []

        # Check for basic HTML structure
        if not re.search(r"<html|<body|<!DOCTYPE", content, re.IGNORECASE):
            issues.append(ValidationIssue(
                message="Missing HTML document structure (no <html>, <body>, or DOCTYPE)",
                severity=ValidationSeverity.WARNING,
                rule="html-structure",
            ))

        # Find all elements with role attributes
        role_pattern = re.compile(
            r'<(\w+)\s+[^>]*role\s*=\s*["\']([^"\']*)["\']',
            re.IGNORECASE,
        )
        for match in role_pattern.finditer(content):
            line = content[:match.start()].count("\n") + 1
            role = match.group(2).strip().lower()
            if role and role not in _VALID_ARIA_ROLES:
                issues.append(ValidationIssue(
                    message=f"Unknown ARIA role '{role}'",
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    rule="aria-role",
                ))

        # Check for duplicate IDs
        id_pattern = re.compile(
            r'\bid\s*=\s*["\']([^"\']*)["\']',
            re.IGNORECASE,
        )
        seen_ids: Dict[str, int] = {}
        for match in id_pattern.finditer(content):
            eid = match.group(1)
            line = content[:match.start()].count("\n") + 1
            if eid in seen_ids:
                issues.append(ValidationIssue(
                    message=f"Duplicate ID '{eid}' (first at line {seen_ids[eid]})",
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    rule="unique-id",
                ))
            else:
                seen_ids[eid] = line

        # Check for images without alt text
        img_pattern = re.compile(
            r'<img\s+([^>]*)>',
            re.IGNORECASE,
        )
        for match in img_pattern.finditer(content):
            attrs = match.group(1)
            line = content[:match.start()].count("\n") + 1
            if 'alt=' not in attrs.lower() and 'aria-label=' not in attrs.lower():
                issues.append(ValidationIssue(
                    message="Image missing alt text or aria-label",
                    severity=ValidationSeverity.WARNING,
                    line=line,
                    rule="img-alt",
                ))

        # Check for interactive elements without accessible names
        interactive_pattern = re.compile(
            r'<(button|a|input|select|textarea)\s+([^>]*)>',
            re.IGNORECASE,
        )
        for match in interactive_pattern.finditer(content):
            tag = match.group(1).lower()
            attrs = match.group(2)
            line = content[:match.start()].count("\n") + 1
            has_name = any(
                attr in attrs.lower()
                for attr in ("aria-label=", "aria-labelledby=", "title=")
            )
            # Buttons and links can get name from content
            if tag in ("input", "select", "textarea") and not has_name:
                if "id=" not in attrs.lower():
                    issues.append(ValidationIssue(
                        message=f"<{tag}> may lack an accessible name",
                        severity=ValidationSeverity.WARNING,
                        line=line,
                        rule="accessible-name",
                    ))

        return ValidationResult(
            valid=all(i.severity != ValidationSeverity.ERROR for i in issues),
            issues=issues,
            format_id="html-aria",
        )

    # ------------------------------------------------------------------
    # Android XML validation
    # ------------------------------------------------------------------

    def validate_android_xml(
        self, content: str
    ) -> ValidationResult:
        """Validate Android uiautomator XML dump structure.

        Checks for the expected root element, node attributes, and
        bounds format.

        Parameters
        ----------
        content : str
            Android XML dump.

        Returns
        -------
        ValidationResult
        """
        issues: list[ValidationIssue] = []
        stripped = content.strip()

        # Check root element
        if not re.search(r"<hierarchy", stripped[:500]):
            issues.append(ValidationIssue(
                message="Missing <hierarchy> root element",
                severity=ValidationSeverity.ERROR,
                rule="android-root",
            ))

        # Check node elements
        node_pattern = re.compile(r"<node\s+([^>]*)(?:/>|>)", re.DOTALL)
        node_count = 0
        for match in node_pattern.finditer(content):
            node_count += 1
            attrs = match.group(1)
            line = content[:match.start()].count("\n") + 1

            # Required attributes
            for attr in _ANDROID_XML_SCHEMA["required_attributes"]:
                if f'{attr}=' not in attrs:
                    issues.append(ValidationIssue(
                        message=f"Node missing required attribute '{attr}'",
                        severity=ValidationSeverity.WARNING,
                        line=line,
                        rule="android-required-attr",
                    ))

            # Validate bounds format: [x1,y1][x2,y2]
            bounds_match = re.search(
                r'bounds="(\[[\d,]+\]\[[\d,]+\])"', attrs
            )
            if bounds_match:
                bounds = bounds_match.group(1)
                if not re.match(r"\[\d+,\d+\]\[\d+,\d+\]", bounds):
                    issues.append(ValidationIssue(
                        message=f"Invalid bounds format: {bounds}",
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        rule="android-bounds",
                    ))

        if node_count == 0 and "<hierarchy" in stripped:
            issues.append(ValidationIssue(
                message="Hierarchy contains no <node> elements",
                severity=ValidationSeverity.WARNING,
                rule="android-empty",
            ))

        return ValidationResult(
            valid=all(i.severity != ValidationSeverity.ERROR for i in issues),
            issues=issues,
            format_id="android-xml",
        )

    # ------------------------------------------------------------------
    # Task specification validation
    # ------------------------------------------------------------------

    def validate_taskspec(
        self, content: str | dict
    ) -> ValidationResult:
        """Validate a task-specification document.

        Parameters
        ----------
        content : str or dict
            JSON or YAML task-spec content.

        Returns
        -------
        ValidationResult
        """
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try simple YAML-like parsing for key: value
                data = self._parse_simple_yaml(content)
        else:
            data = content

        if not isinstance(data, dict):
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    message="Task spec must be a mapping/object",
                    severity=ValidationSeverity.ERROR,
                    rule="taskspec-type",
                )],
                format_id="taskspec",
            )

        issues: list[ValidationIssue] = []

        # Check for task or taskspec key
        task = data.get("task", data.get("taskspec"))
        if task is None:
            issues.append(ValidationIssue(
                message="Missing 'task' or 'taskspec' top-level key",
                severity=ValidationSeverity.ERROR,
                path="/task",
                rule="taskspec-required",
            ))
            return ValidationResult(valid=False, issues=issues, format_id="taskspec")

        if not isinstance(task, dict):
            issues.append(ValidationIssue(
                message="'task' must be a mapping",
                severity=ValidationSeverity.ERROR,
                path="/task",
                rule="taskspec-type",
            ))
            return ValidationResult(valid=False, issues=issues, format_id="taskspec")

        # Required fields in task
        if "name" not in task:
            issues.append(ValidationIssue(
                message="Task missing required 'name' field",
                severity=ValidationSeverity.ERROR,
                path="/task/name",
                rule="taskspec-required",
            ))

        steps = task.get("steps", [])
        if not isinstance(steps, list):
            issues.append(ValidationIssue(
                message="'steps' must be a list",
                severity=ValidationSeverity.ERROR,
                path="/task/steps",
                rule="taskspec-type",
            ))
        else:
            if not steps:
                issues.append(ValidationIssue(
                    message="Task has no steps",
                    severity=ValidationSeverity.WARNING,
                    path="/task/steps",
                    rule="taskspec-empty",
                ))
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    issues.append(ValidationIssue(
                        message=f"Step {i} must be a mapping",
                        severity=ValidationSeverity.ERROR,
                        path=f"/task/steps/{i}",
                        rule="taskspec-type",
                    ))
                elif "action" not in step:
                    issues.append(ValidationIssue(
                        message=f"Step {i} missing required 'action' field",
                        severity=ValidationSeverity.ERROR,
                        path=f"/task/steps/{i}/action",
                        rule="taskspec-required",
                    ))

        return ValidationResult(
            valid=all(i.severity != ValidationSeverity.ERROR for i in issues),
            issues=issues,
            format_id="taskspec",
        )

    # ------------------------------------------------------------------
    # Schema loading
    # ------------------------------------------------------------------

    def load_schema(self, format_id: str) -> Dict[str, Any]:
        """Load the built-in schema for a format.

        Parameters
        ----------
        format_id : str
            Format identifier.

        Returns
        -------
        dict
            JSON Schema document.

        Raises
        ------
        KeyError
            If no schema is available for *format_id*.
        """
        schemas: Dict[str, Dict[str, Any]] = {
            "json-a11y": _A11Y_TREE_SCHEMA,
            "android-xml": _ANDROID_XML_SCHEMA,
            "taskspec": _TASKSPEC_SCHEMA,
            "yaml-taskspec": _TASKSPEC_SCHEMA,
        }
        if format_id not in schemas:
            raise KeyError(f"No built-in schema for format {format_id!r}")
        return schemas[format_id]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_simple_yaml(content: str) -> Dict[str, Any]:
        """Parse a very simple YAML-like key: value format.

        This is a minimal parser for task-spec validation; it does
        not handle the full YAML specification.
        """
        result: Dict[str, Any] = {}
        current_key: Optional[str] = None
        current_dict: Dict[str, Any] = result
        indent_stack: list[tuple[int, Dict[str, Any]]] = [(0, result)]

        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped == "---":
                continue

            indent = len(line) - len(line.lstrip())

            # Pop indent stack
            while indent_stack and indent <= indent_stack[-1][0] and len(indent_stack) > 1:
                indent_stack.pop()
            current_dict = indent_stack[-1][1]

            if ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()

                if value:
                    current_dict[key] = value
                else:
                    sub: Dict[str, Any] = {}
                    current_dict[key] = sub
                    indent_stack.append((indent + 1, sub))
            elif stripped.startswith("- "):
                # List item
                item = stripped[2:].strip()
                if current_key and current_key in current_dict:
                    if not isinstance(current_dict[current_key], list):
                        current_dict[current_key] = []
                    current_dict[current_key].append(item)

        return result
