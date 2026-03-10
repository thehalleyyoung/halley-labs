"""
usability_oracle.simulation.goms — GOMS (Goals, Operators, Methods, Selection) model.

Implements the GOMS family of cognitive task analysis models.  A GOMS model
describes a user's task knowledge as a hierarchy of Goals achieved by Methods
composed of Operators, with Selection Rules choosing among alternative Methods.

This module supports:
- GOMS goal hierarchy construction and traversal
- Execution time prediction via method decomposition
- Critical path analysis (CPM-GOMS / PERT)
- Bottleneck identification
- Natural-language-like GOMS specification parsing

References:
    Card, S. K., Moran, T. P., & Newell, A. (1983). *The Psychology of
        Human-Computer Interaction*. Erlbaum. (Ch. 5-6.)
    John, B. E., & Kieras, D. E. (1996). The GOMS family of user
        interface analysis techniques. *ACM TOCHI*, 3(4), 320-351.
    John, B. E. (1990). Extensions of GOMS analyses to expert performance
        requiring perception of dynamic visual and auditory information.
        *Proceedings of CHI '90*, 107-115.
    Gray, W. D., John, B. E., & Atwood, M. E. (1993). Project Ernestine.
        *Proceedings of CHI '93*, 353-360.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Resource types (for CPM-GOMS)
# ═══════════════════════════════════════════════════════════════════════════

@unique
class ResourceType(Enum):
    """Cognitive/motor/perceptual resource types for CPM-GOMS.

    Reference: John (1990), CPM-GOMS resource classification.
    """
    COGNITIVE = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    EYE = auto()
    EAR = auto()
    VOCAL = auto()
    SYSTEM = auto()

    @property
    def is_motor(self) -> bool:
        return self in {ResourceType.LEFT_HAND, ResourceType.RIGHT_HAND, ResourceType.VOCAL}

    @property
    def is_perceptual(self) -> bool:
        return self in {ResourceType.EYE, ResourceType.EAR}


# ═══════════════════════════════════════════════════════════════════════════
# Core GOMS dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Operator:
    """A primitive operator in the GOMS model.

    An Operator is an elementary action that cannot be decomposed further.
    Its duration is typically derived from the KLM or empirical data.

    Attributes:
        name: Operator identifier (e.g., 'press-key', 'move-cursor').
        duration: Expected execution time (seconds).
        resource: Resource required for this operator.
        variance: Duration variance for stochastic modeling.
        description: Human-readable description.
        metadata: Additional properties.
    """
    name: str = ""
    duration: float = 0.0
    resource: ResourceType = ResourceType.COGNITIVE
    variance: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def std_dev(self) -> float:
        return self.variance ** 0.5 if self.variance > 0 else 0.0


@dataclass
class Goal:
    """A goal in the GOMS hierarchy.

    Goals can have sub-goals, forming a tree.  A goal is achieved by
    executing one of its methods (selected by a selection rule).

    Attributes:
        name: Goal identifier.
        description: Human-readable goal description.
        subgoals: Ordered list of sub-goals.
        completion_condition: Optional predicate for goal satisfaction.
        is_parallel: Whether sub-goals can execute in parallel (CPM-GOMS).
        metadata: Additional properties.
    """
    name: str = ""
    description: str = ""
    subgoals: List[Goal] = field(default_factory=list)
    completion_condition: Optional[Callable[..., bool]] = None
    is_parallel: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.subgoals) == 0

    @property
    def depth(self) -> int:
        if not self.subgoals:
            return 0
        return 1 + max(sg.depth for sg in self.subgoals)

    @property
    def n_descendants(self) -> int:
        total = len(self.subgoals)
        for sg in self.subgoals:
            total += sg.n_descendants
        return total

    def flatten(self) -> List[Goal]:
        """Return all goals in pre-order traversal."""
        result = [self]
        for sg in self.subgoals:
            result.extend(sg.flatten())
        return result


@dataclass
class Method:
    """A method that achieves a goal.

    A method is an ordered sequence of operators and sub-goal invocations.
    Multiple methods may exist for the same goal; selection rules choose
    which method to use based on context.

    Attributes:
        name: Method identifier.
        goal_name: The goal this method achieves.
        steps: Ordered list of (step_type, step_name) pairs.
            step_type is 'operator' or 'goal'.
        operators: Operator definitions used in this method.
        description: Human-readable description.
        applicable_condition: When this method is applicable.
        metadata: Additional properties.
    """
    name: str = ""
    goal_name: str = ""
    steps: List[Tuple[str, str]] = field(default_factory=list)
    operators: Dict[str, Operator] = field(default_factory=dict)
    description: str = ""
    applicable_condition: Optional[Callable[..., bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def operator_steps(self) -> List[Tuple[str, str]]:
        return [(t, n) for t, n in self.steps if t == "operator"]

    @property
    def subgoal_steps(self) -> List[Tuple[str, str]]:
        return [(t, n) for t, n in self.steps if t == "goal"]


@dataclass
class SelectionRule:
    """A rule that selects a method based on context.

    When multiple methods exist for a goal, selection rules determine
    which method to use.

    Attributes:
        goal_name: The goal this rule applies to.
        condition: Callable(context) -> bool.
        method_name: The method to select when condition is True.
        priority: Higher priority rules are checked first.
        description: Human-readable rule description.
    """
    goal_name: str = ""
    condition: Callable[..., bool] = field(default=lambda ctx: True)
    method_name: str = ""
    priority: int = 0
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Standard operator library
# ═══════════════════════════════════════════════════════════════════════════

# Published operator durations from Card, Moran & Newell (1983)
STANDARD_OPERATORS: Dict[str, Operator] = {
    "press-key": Operator(
        name="press-key", duration=0.280, resource=ResourceType.RIGHT_HAND,
        variance=0.012, description="Press a single key (average typist)",
    ),
    "press-key-expert": Operator(
        name="press-key-expert", duration=0.120, resource=ResourceType.RIGHT_HAND,
        variance=0.004, description="Press a key (expert typist, Card et al. 1983)",
    ),
    "move-cursor": Operator(
        name="move-cursor", duration=1.100, resource=ResourceType.RIGHT_HAND,
        variance=0.090, description="Point to target (Fitts' law average, Card et al. 1983)",
    ),
    "click-mouse": Operator(
        name="click-mouse", duration=0.100, resource=ResourceType.RIGHT_HAND,
        variance=0.002, description="Mouse button press",
    ),
    "home-to-mouse": Operator(
        name="home-to-mouse", duration=0.400, resource=ResourceType.RIGHT_HAND,
        variance=0.016, description="Move hand from keyboard to mouse (Card et al. 1983)",
    ),
    "home-to-keyboard": Operator(
        name="home-to-keyboard", duration=0.400, resource=ResourceType.RIGHT_HAND,
        variance=0.016, description="Move hand from mouse to keyboard",
    ),
    "mental-preparation": Operator(
        name="mental-preparation", duration=1.350, resource=ResourceType.COGNITIVE,
        variance=0.182, description="Mental preparation (Card et al. 1983, M operator)",
    ),
    "perceive-change": Operator(
        name="perceive-change", duration=0.100, resource=ResourceType.EYE,
        variance=0.001, description="Perceive a visual change",
    ),
    "eye-fixation": Operator(
        name="eye-fixation", duration=0.230, resource=ResourceType.EYE,
        variance=0.005, description="Single eye fixation (Rayner 1998, avg ~230ms)",
    ),
    "verify-result": Operator(
        name="verify-result", duration=1.200, resource=ResourceType.COGNITIVE,
        variance=0.144, description="Verify output is correct (cognitive check)",
    ),
    "system-response": Operator(
        name="system-response", duration=0.0, resource=ResourceType.SYSTEM,
        variance=0.0, description="Wait for system response (variable)",
    ),
    "read-word": Operator(
        name="read-word", duration=0.260, resource=ResourceType.EYE,
        variance=0.007, description="Read a single word (avg fixation, Rayner 1998)",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# GOMSModel
# ═══════════════════════════════════════════════════════════════════════════

class GOMSModel:
    """GOMS model for task analysis and execution time prediction.

    A GOMS model contains:
    - A hierarchy of Goals
    - Methods that achieve each Goal
    - Selection Rules that choose among alternative Methods
    - Operators that are the primitive actions

    Usage::

        model = GOMSModel()
        model.add_goal(Goal(name="edit-document", subgoals=[...]))
        model.add_method(Method(name="delete-word-method", ...))
        time = model.predict_execution_time("edit-document")
    """

    def __init__(self) -> None:
        self._goals: Dict[str, Goal] = {}
        self._methods: Dict[str, List[Method]] = {}  # goal_name -> [methods]
        self._selection_rules: Dict[str, List[SelectionRule]] = {}
        self._operators: Dict[str, Operator] = dict(STANDARD_OPERATORS)
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def add_goal(self, goal: Goal) -> None:
        """Register a goal."""
        self._goals[goal.name] = goal

    def add_method(self, method: Method) -> None:
        """Register a method for a goal."""
        if method.goal_name not in self._methods:
            self._methods[method.goal_name] = []
        self._methods[method.goal_name].append(method)
        # Register method-local operators
        for op_name, op in method.operators.items():
            self._operators[op_name] = op

    def add_selection_rule(self, rule: SelectionRule) -> None:
        """Register a selection rule."""
        if rule.goal_name not in self._selection_rules:
            self._selection_rules[rule.goal_name] = []
        self._selection_rules[rule.goal_name].append(rule)
        # Sort by priority (descending)
        self._selection_rules[rule.goal_name].sort(key=lambda r: -r.priority)

    def add_operator(self, operator: Operator) -> None:
        """Register a custom operator."""
        self._operators[operator.name] = operator

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the current context for selection rule evaluation."""
        self._context = context

    # ------------------------------------------------------------------
    # Method selection
    # ------------------------------------------------------------------

    def _select_method(self, goal_name: str) -> Optional[Method]:
        """Select the appropriate method for a goal using selection rules."""
        methods = self._methods.get(goal_name, [])
        if not methods:
            return None
        if len(methods) == 1:
            return methods[0]

        # Check selection rules
        rules = self._selection_rules.get(goal_name, [])
        for rule in rules:
            try:
                if rule.condition(self._context):
                    for m in methods:
                        if m.name == rule.method_name:
                            return m
            except Exception:
                continue

        # Default: return first method
        return methods[0]

    # ------------------------------------------------------------------
    # Execution time prediction
    # ------------------------------------------------------------------

    def predict_execution_time(
        self,
        goal_name: str,
        depth: int = 0,
        max_depth: int = 50,
    ) -> float:
        """Predict total execution time by traversing the goal hierarchy.

        Time is computed recursively:
        - For a goal with a method: sum of step times
        - For an operator step: operator duration
        - For a sub-goal step: recursively predict

        For parallel goals (CPM-GOMS): take max instead of sum.

        Reference: Card et al. (1983), Ch. 5.
        """
        if depth > max_depth:
            return 0.0

        goal = self._goals.get(goal_name)
        method = self._select_method(goal_name)

        if method is not None:
            return self._method_time(method, depth, max_depth)

        # No method defined — check sub-goals from goal object
        if goal is not None and goal.subgoals:
            times = [
                self.predict_execution_time(sg.name, depth + 1, max_depth)
                for sg in goal.subgoals
            ]
            if goal.is_parallel:
                return max(times) if times else 0.0
            return sum(times)

        return 0.0

    def _method_time(self, method: Method, depth: int, max_depth: int) -> float:
        """Compute execution time for a method."""
        total = 0.0
        for step_type, step_name in method.steps:
            if step_type == "operator":
                op = method.operators.get(step_name) or self._operators.get(step_name)
                if op:
                    total += op.duration
            elif step_type == "goal":
                total += self.predict_execution_time(step_name, depth + 1, max_depth)
        return total

    def predict_with_variance(
        self,
        goal_name: str,
        depth: int = 0,
        max_depth: int = 50,
    ) -> Tuple[float, float]:
        """Predict execution time with variance propagation.

        Returns (mean_time, total_variance) using the assumption that
        operator times are independent.

        Reference: PERT analysis (Clark, 1962).
        """
        if depth > max_depth:
            return 0.0, 0.0

        method = self._select_method(goal_name)
        goal = self._goals.get(goal_name)

        if method is not None:
            return self._method_time_with_variance(method, depth, max_depth)

        if goal is not None and goal.subgoals:
            means_vars = [
                self.predict_with_variance(sg.name, depth + 1, max_depth)
                for sg in goal.subgoals
            ]
            means = [mv[0] for mv in means_vars]
            variances = [mv[1] for mv in means_vars]

            if goal.is_parallel:
                # For parallel: time ≈ max(means); variance is approximated
                total_mean = max(means) if means else 0.0
                max_idx = means.index(max(means)) if means else 0
                total_var = variances[max_idx] if variances else 0.0
                return total_mean, total_var
            return sum(means), sum(variances)

        return 0.0, 0.0

    def _method_time_with_variance(
        self, method: Method, depth: int, max_depth: int,
    ) -> Tuple[float, float]:
        """Compute time with variance for a method."""
        total_mean = 0.0
        total_var = 0.0
        for step_type, step_name in method.steps:
            if step_type == "operator":
                op = method.operators.get(step_name) or self._operators.get(step_name)
                if op:
                    total_mean += op.duration
                    total_var += op.variance
            elif step_type == "goal":
                m, v = self.predict_with_variance(step_name, depth + 1, max_depth)
                total_mean += m
                total_var += v
        return total_mean, total_var

    # ------------------------------------------------------------------
    # Critical path analysis (CPM-GOMS / PERT)
    # ------------------------------------------------------------------

    def critical_path(self, goal_name: str) -> List[Tuple[str, float]]:
        """Find the critical path through the goal tree.

        The critical path is the longest path from root to any leaf,
        determining the minimum possible completion time.

        Returns:
            List of (step_description, cumulative_time) pairs along the
            critical path.

        Reference: John (1990), CPM-GOMS.
        """
        paths = self._enumerate_paths(goal_name)
        if not paths:
            return []
        # Find the path with maximum total time
        return max(paths, key=lambda p: p[-1][1] if p else 0.0)

    def _enumerate_paths(
        self,
        goal_name: str,
        prefix: Optional[List[Tuple[str, float]]] = None,
        cumulative_time: float = 0.0,
    ) -> List[List[Tuple[str, float]]]:
        """Enumerate all root-to-leaf paths with cumulative times."""
        if prefix is None:
            prefix = []

        method = self._select_method(goal_name)
        goal = self._goals.get(goal_name)

        if method is not None:
            path = list(prefix)
            current_time = cumulative_time
            all_paths: List[List[Tuple[str, float]]] = []

            for step_type, step_name in method.steps:
                if step_type == "operator":
                    op = method.operators.get(step_name) or self._operators.get(step_name)
                    if op:
                        current_time += op.duration
                        path.append((f"[{goal_name}] {step_name}", current_time))
                elif step_type == "goal":
                    sub_paths = self._enumerate_paths(step_name, list(path), current_time)
                    if sub_paths:
                        for sp in sub_paths:
                            all_paths.append(sp)
                            if sp:
                                current_time = max(current_time, sp[-1][1])
                        path = list(max(sub_paths, key=lambda p: p[-1][1] if p else 0.0))

            if not all_paths:
                all_paths = [path]
            return all_paths

        if goal is not None and goal.subgoals:
            if goal.is_parallel:
                # Parallel: each sub-goal is an independent path
                all_paths = []
                for sg in goal.subgoals:
                    sub_paths = self._enumerate_paths(sg.name, list(prefix), cumulative_time)
                    all_paths.extend(sub_paths)
                return all_paths if all_paths else [prefix]
            else:
                # Sequential: chain sub-goals
                current_paths = [list(prefix)]
                ct = cumulative_time
                for sg in goal.subgoals:
                    new_paths = []
                    for cp in current_paths:
                        sub_paths = self._enumerate_paths(sg.name, cp, ct)
                        if sub_paths:
                            new_paths.extend(sub_paths)
                            ct = max(sp[-1][1] for sp in sub_paths if sp)
                        else:
                            new_paths.append(cp)
                    current_paths = new_paths
                return current_paths

        return [prefix] if prefix else []

    # ------------------------------------------------------------------
    # Bottleneck analysis
    # ------------------------------------------------------------------

    def identify_bottleneck_goals(
        self,
        goal_name: str,
        depth: int = 0,
        max_depth: int = 50,
    ) -> List[Dict[str, Any]]:
        """Identify goals with the highest time contribution.

        Returns a list of goals sorted by their contribution to total time,
        enabling targeted design improvements.

        Reference: Gray, John & Atwood (1993), Project Ernestine methodology.
        """
        total_time = self.predict_execution_time(goal_name)
        if total_time <= 0:
            return []

        contributions: List[Dict[str, Any]] = []
        self._collect_contributions(goal_name, contributions, depth, max_depth)

        # Sort by time contribution (descending)
        contributions.sort(key=lambda c: -c["time"])

        # Add percentage
        for c in contributions:
            c["pct_of_total"] = (c["time"] / total_time * 100.0) if total_time > 0 else 0.0

        return contributions

    def _collect_contributions(
        self,
        goal_name: str,
        contributions: List[Dict[str, Any]],
        depth: int,
        max_depth: int,
    ) -> None:
        """Recursively collect time contributions for each goal/operator."""
        if depth > max_depth:
            return

        goal_time = self.predict_execution_time(goal_name)
        contributions.append({
            "goal_name": goal_name,
            "time": goal_time,
            "depth": depth,
            "type": "goal",
        })

        method = self._select_method(goal_name)
        if method:
            for step_type, step_name in method.steps:
                if step_type == "operator":
                    op = method.operators.get(step_name) or self._operators.get(step_name)
                    if op:
                        contributions.append({
                            "goal_name": f"{goal_name}/{step_name}",
                            "time": op.duration,
                            "depth": depth + 1,
                            "type": "operator",
                            "resource": op.resource.name,
                        })
                elif step_type == "goal":
                    self._collect_contributions(step_name, contributions, depth + 1, max_depth)

        goal = self._goals.get(goal_name)
        if goal and not method:
            for sg in goal.subgoals:
                self._collect_contributions(sg.name, contributions, depth + 1, max_depth)

    # ------------------------------------------------------------------
    # GOMS specification parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_goms_spec(text: str) -> Tuple[Dict[str, Goal], Dict[str, List[Method]]]:
        """Parse a natural-language-like GOMS specification.

        Format::

            GOAL: edit-document
              GOAL: select-text
                METHOD: select-by-mouse
                  STEP: move-cursor (operator, 1.100s)
                  STEP: click-mouse (operator, 0.100s)
                  STEP: drag-to-end (operator, 1.500s)
              GOAL: delete-selected
                METHOD: delete-key-method
                  STEP: press-key (operator, 0.280s)

        Returns:
            Tuple of (goals_dict, methods_dict).
        """
        goals: Dict[str, Goal] = {}
        methods: Dict[str, List[Method]] = {}
        goal_stack: List[Goal] = []
        current_method: Optional[Method] = None

        for line in text.strip().split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())

            # GOAL line
            goal_match = re.match(r"GOAL:\s*(.+)", stripped)
            if goal_match:
                goal_name = goal_match.group(1).strip()
                goal = Goal(name=goal_name)
                goals[goal_name] = goal

                # Determine parent based on indentation
                while goal_stack and indent <= goal_stack[-1].metadata.get("_indent", 0):
                    goal_stack.pop()

                if goal_stack:
                    goal_stack[-1].subgoals.append(goal)

                goal.metadata["_indent"] = indent
                goal_stack.append(goal)
                current_method = None
                continue

            # METHOD line
            method_match = re.match(r"METHOD:\s*(.+)", stripped)
            if method_match:
                method_name = method_match.group(1).strip()
                parent_goal = goal_stack[-1].name if goal_stack else ""
                current_method = Method(name=method_name, goal_name=parent_goal)
                if parent_goal not in methods:
                    methods[parent_goal] = []
                methods[parent_goal].append(current_method)
                continue

            # STEP line
            step_match = re.match(
                r"STEP:\s*(\S+)\s*\((\w+)(?:,\s*([\d.]+)s?)?\)", stripped
            )
            if step_match and current_method is not None:
                step_name = step_match.group(1)
                step_type = step_match.group(2)  # 'operator' or 'goal'
                duration_str = step_match.group(3)

                if step_type == "operator":
                    duration = float(duration_str) if duration_str else 0.0
                    op = Operator(name=step_name, duration=duration)
                    current_method.operators[step_name] = op
                    current_method.steps.append(("operator", step_name))
                elif step_type == "goal":
                    current_method.steps.append(("goal", step_name))
                continue

        return goals, methods

    # ------------------------------------------------------------------
    # Model construction from spec
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, text: str) -> GOMSModel:
        """Build a GOMSModel from a GOMS specification string."""
        model = cls()
        goals, methods = cls.parse_goms_spec(text)
        for goal in goals.values():
            model.add_goal(goal)
        for goal_methods in methods.values():
            for method in goal_methods:
                model.add_method(method)
        return model

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self, goal_name: str) -> str:
        """Human-readable summary of GOMS analysis for a goal."""
        total = self.predict_execution_time(goal_name)
        mean, var = self.predict_with_variance(goal_name)
        bottlenecks = self.identify_bottleneck_goals(goal_name)

        lines = [f"GOMS Analysis: {goal_name}"]
        lines.append(f"  Predicted time: {total:.3f}s")
        lines.append(f"  Std deviation: {var ** 0.5:.3f}s")
        if bottlenecks:
            lines.append(f"  Top bottleneck: {bottlenecks[0]['goal_name']} "
                         f"({bottlenecks[0]['time']:.3f}s, "
                         f"{bottlenecks[0].get('pct_of_total', 0):.1f}%)")
        lines.append(f"  Total goals: {len(self._goals)}")
        lines.append(f"  Total methods: {sum(len(m) for m in self._methods.values())}")
        return "\n".join(lines)

    @property
    def goals(self) -> Dict[str, Goal]:
        return dict(self._goals)

    @property
    def operators(self) -> Dict[str, Operator]:
        return dict(self._operators)

    @property
    def n_goals(self) -> int:
        return len(self._goals)

    @property
    def n_methods(self) -> int:
        return sum(len(m) for m in self._methods.values())
