"""
usability_oracle.taskspec — Task flow specification DSL and tooling.

Provides a YAML-based domain-specific language for defining UI tasks,
along with recording, inference, validation, and template facilities.

Key classes
-----------
- :class:`TaskStep` — a single atomic user action
- :class:`TaskFlow` — ordered sequence of steps comprising a task
- :class:`TaskSpec` — full specification (multiple flows + metadata)
- :class:`TaskGraph` — dependency graph over steps
- :class:`TaskDSLParser` — YAML DSL parser
- :class:`TaskRecorder` — record steps from event logs
- :class:`TaskInferrer` — auto-infer tasks from accessibility trees
- :class:`TaskValidator` — validate specs against live trees
- :class:`TaskTemplates` — library of pre-built task templates
"""

from __future__ import annotations

from usability_oracle.taskspec.models import (
    TaskStep,
    TaskFlow,
    TaskSpec,
    TaskGraph,
)
from usability_oracle.taskspec.dsl import TaskDSLParser
from usability_oracle.taskspec.recorder import TaskRecorder
from usability_oracle.taskspec.inference import TaskInferrer
from usability_oracle.taskspec.validator import TaskValidator, ValidationResult, ValidationIssue
from usability_oracle.taskspec.templates import TaskTemplates

__all__ = [
    "TaskStep",
    "TaskFlow",
    "TaskSpec",
    "TaskGraph",
    "TaskDSLParser",
    "TaskRecorder",
    "TaskInferrer",
    "TaskValidator",
    "ValidationResult",
    "ValidationIssue",
    "TaskTemplates",
]
