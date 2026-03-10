"""
usability_oracle.taskspec.validator — Validate task specifications against
live accessibility trees.

The :class:`TaskValidator` checks that:

1. Every target referenced in a :class:`TaskSpec` exists in the tree.
2. All targets are reachable (not hidden, disabled, or obscured).
3. Step preconditions are satisfiable given the flow ordering.
4. Flows are complete (every required postcondition is eventually met).

Results are returned as a :class:`ValidationResult` containing zero or more
:class:`ValidationIssue` objects graded by severity.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from usability_oracle.taskspec.models import TaskFlow, TaskSpec, TaskStep


# ---------------------------------------------------------------------------
# Protocols for accessibility tree interaction
# ---------------------------------------------------------------------------

class AccessibilityNode(Protocol):
    @property
    def node_id(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def children(self) -> Sequence["AccessibilityNode"]: ...

    @property
    def properties(self) -> Dict[str, Any]: ...


class AccessibilityTree(Protocol):
    @property
    def root(self) -> AccessibilityNode: ...

    def find_by_role(self, role: str) -> List[AccessibilityNode]: ...

    def find_by_name(self, name: str) -> List[AccessibilityNode]: ...


# ---------------------------------------------------------------------------
# Severity and issue types
# ---------------------------------------------------------------------------


class IssueSeverity(enum.Enum):
    """Severity levels for validation issues."""

    ERROR = "error"          # task cannot be completed
    WARNING = "warning"      # task may fail under some conditions
    INFO = "info"            # informational observation


class IssueCategory(enum.Enum):
    """Category of a validation issue."""

    TARGET_MISSING = "target_missing"
    TARGET_HIDDEN = "target_hidden"
    TARGET_DISABLED = "target_disabled"
    TARGET_AMBIGUOUS = "target_ambiguous"
    UNREACHABLE = "unreachable"
    PRECONDITION_UNMET = "precondition_unmet"
    POSTCONDITION_MISSING = "postcondition_missing"
    FLOW_INCOMPLETE = "flow_incomplete"
    CYCLE_DETECTED = "cycle_detected"
    TIMEOUT_RISK = "timeout_risk"


# ---------------------------------------------------------------------------
# ValidationIssue & ValidationResult
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: IssueSeverity
    category: IssueCategory
    message: str
    step_id: Optional[str] = None
    flow_id: Optional[str] = None
    target_role: Optional[str] = None
    target_name: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
        }
        if self.step_id:
            d["step_id"] = self.step_id
        if self.flow_id:
            d["flow_id"] = self.flow_id
        if self.target_role:
            d["target_role"] = self.target_role
        if self.target_name:
            d["target_name"] = self.target_name
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class ValidationResult:
    """Aggregate validation result for a :class:`TaskSpec`."""

    spec_id: str
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if there are no ERROR-level issues."""
        return not any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.INFO)

    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"[{status}] spec={self.spec_id}: "
            f"{self.error_count} errors, {self.warning_count} warnings, "
            f"{self.info_count} info"
        )


# ---------------------------------------------------------------------------
# Helpers for tree traversal
# ---------------------------------------------------------------------------

def _collect_all_nodes(node: Any, *, max_depth: int = 20) -> List[Any]:
    """Recursively collect all nodes from a tree."""
    result: List[Any] = [node]
    if max_depth <= 0:
        return result
    for child in getattr(node, "children", []):
        result.extend(_collect_all_nodes(child, max_depth=max_depth - 1))
    return result


def _find_matching_nodes(
    tree: Any, role: str, name: str
) -> List[Any]:
    """Find nodes matching the given role and name."""
    root = getattr(tree, "root", tree)
    all_nodes = _collect_all_nodes(root)
    matches: List[Any] = []
    for node in all_nodes:
        node_role = getattr(node, "role", "").lower()
        node_name = getattr(node, "name", "")
        role_match = (not role) or (node_role == role.lower())
        name_match = (not name) or (node_name == name)
        if role_match and name_match:
            matches.append(node)
    return matches


def _is_node_hidden(node: Any) -> bool:
    """Check whether a node is hidden / not visible."""
    props = getattr(node, "properties", {})
    if props.get("hidden", False):
        return True
    if props.get("aria-hidden", "false").lower() == "true":
        return True
    if props.get("display") == "none":
        return True
    if props.get("visibility") == "hidden":
        return True
    return False


def _is_node_disabled(node: Any) -> bool:
    """Check whether a node is disabled."""
    props = getattr(node, "properties", {})
    return bool(props.get("disabled", False) or props.get("aria-disabled", "false").lower() == "true")


# ---------------------------------------------------------------------------
# TaskValidator
# ---------------------------------------------------------------------------


class TaskValidator:
    """Validate a :class:`TaskSpec` against an accessibility tree.

    Usage::

        validator = TaskValidator()
        result = validator.validate(spec, tree)
        if not result.is_valid:
            for issue in result.errors():
                print(issue.message)
    """

    def validate(self, spec: TaskSpec, tree: Any) -> ValidationResult:
        """Run all validation checks.

        Parameters
        ----------
        spec : TaskSpec
            The task specification to validate.
        tree : AccessibilityTree
            The live accessibility tree to validate against.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult(spec_id=spec.spec_id)

        # Structural validation first
        structural_errors = spec.validate()
        for err_msg in structural_errors:
            result.issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.FLOW_INCOMPLETE,
                message=err_msg,
            ))

        # Tree-dependent checks
        result.issues.extend(self._check_targets_exist(spec, tree))
        result.issues.extend(self._check_reachability(spec, tree))
        result.issues.extend(self._check_preconditions(spec))
        result.issues.extend(self._check_flow_completeness(spec))

        return result

    # -- check: targets exist ------------------------------------------------

    def _check_targets_exist(self, spec: TaskSpec, tree: Any) -> List[ValidationIssue]:
        """Verify that every step's target exists in the tree."""
        issues: List[ValidationIssue] = []
        for flow in spec.flows:
            for step in flow.steps:
                if not step.target_role and not step.target_name:
                    continue  # no target to check (e.g. wait)

                matches = _find_matching_nodes(tree, step.target_role, step.target_name)

                if not matches:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.TARGET_MISSING,
                        message=(
                            f"Target not found: role={step.target_role!r}, "
                            f"name={step.target_name!r}"
                        ),
                        step_id=step.step_id,
                        flow_id=flow.flow_id,
                        target_role=step.target_role,
                        target_name=step.target_name,
                        suggestion=self._suggest_similar_target(tree, step),
                    ))
                elif len(matches) > 1:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.TARGET_AMBIGUOUS,
                        message=(
                            f"Ambiguous target ({len(matches)} matches): "
                            f"role={step.target_role!r}, name={step.target_name!r}. "
                            f"Consider adding a selector."
                        ),
                        step_id=step.step_id,
                        flow_id=flow.flow_id,
                        target_role=step.target_role,
                        target_name=step.target_name,
                    ))

        return issues

    # -- check: reachability -------------------------------------------------

    def _check_reachability(self, spec: TaskSpec, tree: Any) -> List[ValidationIssue]:
        """Check that targets are not hidden or disabled."""
        issues: List[ValidationIssue] = []
        for flow in spec.flows:
            for step in flow.steps:
                if not step.target_role and not step.target_name:
                    continue

                matches = _find_matching_nodes(tree, step.target_role, step.target_name)

                for node in matches:
                    if _is_node_hidden(node):
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.TARGET_HIDDEN,
                            message=(
                                f"Target is hidden: role={step.target_role!r}, "
                                f"name={step.target_name!r}"
                            ),
                            step_id=step.step_id,
                            flow_id=flow.flow_id,
                            target_role=step.target_role,
                            target_name=step.target_name,
                            suggestion="Ensure the target becomes visible before this step.",
                        ))

                    if _is_node_disabled(node) and step.action_type != "verify":
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING
                            if step.optional
                            else IssueSeverity.ERROR,
                            category=IssueCategory.TARGET_DISABLED,
                            message=(
                                f"Target is disabled: role={step.target_role!r}, "
                                f"name={step.target_name!r}"
                            ),
                            step_id=step.step_id,
                            flow_id=flow.flow_id,
                            target_role=step.target_role,
                            target_name=step.target_name,
                            suggestion="Add a precondition or prior step that enables this target.",
                        ))

        return issues

    # -- check: preconditions ------------------------------------------------

    def _check_preconditions(self, spec: TaskSpec) -> List[ValidationIssue]:
        """Check that step preconditions are established by prior postconditions."""
        issues: List[ValidationIssue] = []

        for flow in spec.flows:
            established: Set[str] = set()
            # Include initial state keys as established predicates
            for key, value in spec.initial_state.items():
                established.add(f"{key} == {value}")
                established.add(key)

            for step in flow.steps:
                for pre in step.preconditions:
                    if pre not in established:
                        # Check if a weaker match exists (key-only match)
                        pre_key = pre.split("==")[0].strip() if "==" in pre else pre
                        if pre_key not in established:
                            issues.append(ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                category=IssueCategory.PRECONDITION_UNMET,
                                message=(
                                    f"Precondition {pre!r} for step {step.step_id!r} "
                                    f"not established by prior steps"
                                ),
                                step_id=step.step_id,
                                flow_id=flow.flow_id,
                                suggestion="Add a prior step that establishes this condition.",
                            ))

                # Register postconditions
                for post in step.postconditions:
                    established.add(post)
                    post_key = post.split("==")[0].strip() if "==" in post else post
                    established.add(post_key)

        return issues

    # -- check: flow completeness --------------------------------------------

    def _check_flow_completeness(self, spec: TaskSpec) -> List[ValidationIssue]:
        """Check that success criteria can be satisfied by the flow's steps."""
        issues: List[ValidationIssue] = []

        for flow in spec.flows:
            if not flow.success_criteria:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.FLOW_INCOMPLETE,
                    message=f"Flow {flow.flow_id!r} has no success criteria.",
                    flow_id=flow.flow_id,
                ))
                continue

            # Collect all postconditions from the flow
            all_postconditions: Set[str] = set()
            for step in flow.steps:
                all_postconditions.update(step.postconditions)

            for criterion in flow.success_criteria:
                if criterion not in all_postconditions:
                    # Check for partial matches
                    criterion_key = criterion.split("==")[0].strip() if "==" in criterion else criterion
                    partial_match = any(
                        criterion_key in post for post in all_postconditions
                    )
                    if not partial_match:
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.POSTCONDITION_MISSING,
                            message=(
                                f"Success criterion {criterion!r} not guaranteed by any "
                                f"step postcondition in flow {flow.flow_id!r}"
                            ),
                            flow_id=flow.flow_id,
                            suggestion="Add a postcondition to an appropriate step.",
                        ))

            # Check for steps without any required step after them (dead ends)
            if flow.steps:
                last_required = None
                for step in reversed(flow.steps):
                    if not step.optional:
                        last_required = step
                        break
                if last_required and last_required.action_type == "wait":
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.INFO,
                        category=IssueCategory.TIMEOUT_RISK,
                        message=(
                            f"Flow {flow.flow_id!r} ends with a 'wait' step — "
                            f"consider adding a verification step."
                        ),
                        flow_id=flow.flow_id,
                    ))

        return issues

    # -- suggestion helpers --------------------------------------------------

    def _suggest_similar_target(self, tree: Any, step: TaskStep) -> Optional[str]:
        """Suggest a similar target from the tree if the exact match fails."""
        root = getattr(tree, "root", tree)
        all_nodes = _collect_all_nodes(root)

        candidates: List[Tuple[float, str, str]] = []
        for node in all_nodes:
            node_role = getattr(node, "role", "").lower()
            node_name = getattr(node, "name", "")
            if not node_name:
                continue

            score = 0.0
            if step.target_role and node_role == step.target_role.lower():
                score += 0.5
            if step.target_name and step.target_name.lower() in node_name.lower():
                score += 0.5
            elif step.target_name and node_name.lower() in step.target_name.lower():
                score += 0.3

            if score > 0:
                candidates.append((score, node_role, node_name))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0]
            return f"Did you mean role={best[1]!r}, name={best[2]!r}?"

        return None
