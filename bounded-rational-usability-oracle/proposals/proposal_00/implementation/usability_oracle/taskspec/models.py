"""
usability_oracle.taskspec.models — Core data models for task specifications.

Defines the structural building blocks used throughout the taskspec module:

* :class:`TaskStep` — a single atomic user action (click, type, scroll, …)
* :class:`TaskFlow` — an ordered sequence of :class:`TaskStep` instances
* :class:`TaskSpec` — a full task specification (multiple flows + metadata)
* :class:`TaskGraph` — a dependency DAG over steps with topological utilities

The YAML DSL (:mod:`.dsl`) produces these objects, and every downstream
consumer (cost algebra, validator, recorder) operates on them.
"""

from __future__ import annotations

import copy
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Canonical action types recognised by the oracle
# ---------------------------------------------------------------------------

ACTION_TYPES: frozenset[str] = frozenset(
    {
        "click",
        "type",
        "select",
        "scroll",
        "navigate",
        "wait",
        "verify",
        "drag",
        "hover",
        "double_click",
        "right_click",
        "key_press",
    }
)


# ---------------------------------------------------------------------------
# TaskStep
# ---------------------------------------------------------------------------


@dataclass
class TaskStep:
    """A single atomic user action within a task flow.

    Parameters
    ----------
    step_id : str
        Unique identifier for this step.
    action_type : str
        One of the canonical action types (click, type, select, …).
    target_role : str
        ARIA / accessibility role of the target widget (e.g. ``"button"``).
    target_name : str
        Accessible name / label of the target widget.
    target_selector : str | None
        Optional CSS / XPath selector for disambiguation.
    input_value : str | None
        Payload for *type* / *select* actions.
    preconditions : list[str]
        Logical predicates that must hold before this step executes.
    postconditions : list[str]
        Logical predicates that must hold after this step executes.
    optional : bool
        Whether the step may be skipped without failing the task.
    description : str
        Human-readable description of the step.
    timeout : float | None
        Maximum wall-clock time (seconds) for this step.
    depends_on : list[str]
        Step IDs that must be completed before this step.
    metadata : dict
        Arbitrary key-value metadata.
    """

    step_id: str = ""
    action_type: str = "click"
    target_role: str = ""
    target_name: str = ""
    target_selector: Optional[str] = None
    input_value: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    optional: bool = False
    description: str = ""
    timeout: Optional[float] = None
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            self.step_id = f"step-{uuid.uuid4().hex[:8]}"
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"Unknown action_type {self.action_type!r}. "
                f"Must be one of {sorted(ACTION_TYPES)}."
            )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        d: Dict[str, Any] = {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "target_role": self.target_role,
            "target_name": self.target_name,
            "optional": self.optional,
            "description": self.description,
        }
        if self.target_selector is not None:
            d["target_selector"] = self.target_selector
        if self.input_value is not None:
            d["input_value"] = self.input_value
        if self.preconditions:
            d["preconditions"] = list(self.preconditions)
        if self.postconditions:
            d["postconditions"] = list(self.postconditions)
        if self.timeout is not None:
            d["timeout"] = self.timeout
        if self.depends_on:
            d["depends_on"] = list(self.depends_on)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStep":
        """Deserialise from a plain dictionary."""
        return cls(
            step_id=data.get("step_id", ""),
            action_type=data.get("action_type", "click"),
            target_role=data.get("target_role", ""),
            target_name=data.get("target_name", ""),
            target_selector=data.get("target_selector"),
            input_value=data.get("input_value"),
            preconditions=list(data.get("preconditions", [])),
            postconditions=list(data.get("postconditions", [])),
            optional=bool(data.get("optional", False)),
            description=data.get("description", ""),
            timeout=data.get("timeout"),
            depends_on=list(data.get("depends_on", [])),
            metadata=dict(data.get("metadata", {})),
        )

    # -- helpers -------------------------------------------------------------

    @property
    def is_input_action(self) -> bool:
        """Return *True* if the step requires user-provided input."""
        return self.action_type in {"type", "select"}

    @property
    def target_descriptor(self) -> str:
        """Human-readable target description."""
        parts = []
        if self.target_role:
            parts.append(self.target_role)
        if self.target_name:
            parts.append(f'"{self.target_name}"')
        return " ".join(parts) if parts else "(unknown target)"


# ---------------------------------------------------------------------------
# TaskFlow
# ---------------------------------------------------------------------------


@dataclass
class TaskFlow:
    """An ordered sequence of :class:`TaskStep` instances forming a single task flow.

    Parameters
    ----------
    flow_id : str
        Unique identifier for this flow.
    name : str
        Human-readable name.
    steps : list[TaskStep]
        Ordered list of steps.
    success_criteria : list[str]
        Predicates that define successful completion.
    max_time : float | None
        Maximum wall-clock time budget for the whole flow (seconds).
    description : str
        Prose description of the flow.
    metadata : dict
        Arbitrary key-value metadata.
    """

    flow_id: str = ""
    name: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    max_time: Optional[float] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.flow_id:
            self.flow_id = f"flow-{uuid.uuid4().hex[:8]}"

    # -- queries -------------------------------------------------------------

    def step_ids(self) -> List[str]:
        return [s.step_id for s in self.steps]

    def get_step(self, step_id: str) -> Optional[TaskStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def required_steps(self) -> List[TaskStep]:
        """Return only non-optional steps."""
        return [s for s in self.steps if not s.optional]

    def input_steps(self) -> List[TaskStep]:
        """Return steps that require user input."""
        return [s for s in self.steps if s.is_input_action]

    def action_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for s in self.steps:
            counts[s.action_type] += 1
        return dict(counts)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "flow_id": self.flow_id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.success_criteria:
            d["success_criteria"] = list(self.success_criteria)
        if self.max_time is not None:
            d["max_time"] = self.max_time
        if self.description:
            d["description"] = self.description
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskFlow":
        return cls(
            flow_id=data.get("flow_id", ""),
            name=data.get("name", ""),
            steps=[TaskStep.from_dict(s) for s in data.get("steps", [])],
            success_criteria=list(data.get("success_criteria", [])),
            max_time=data.get("max_time"),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Full task specification: one or more :class:`TaskFlow` instances
    with shared context, initial state, and metadata.

    Parameters
    ----------
    spec_id : str
        Unique specification identifier.
    name : str
        Human-readable task name.
    description : str
        Prose description.
    flows : list[TaskFlow]
        Ordered list of alternative or sequential task flows.
    initial_state : dict
        Key-value pairs describing the assumed starting UI state.
    metadata : dict
        Arbitrary metadata (author, version, tags, …).
    """

    spec_id: str = ""
    name: str = ""
    description: str = ""
    flows: List[TaskFlow] = field(default_factory=list)
    initial_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.spec_id:
            self.spec_id = f"spec-{uuid.uuid4().hex[:8]}"

    # -- queries -------------------------------------------------------------

    def get_flow(self, flow_id: str) -> Optional[TaskFlow]:
        """Look up a flow by its ID.  Return *None* if not found."""
        for f in self.flows:
            if f.flow_id == flow_id:
                return f
        return None

    def total_steps(self) -> int:
        """Total number of steps across all flows."""
        return sum(len(f.steps) for f in self.flows)

    def all_steps(self) -> Iterator[TaskStep]:
        """Iterate over every step in every flow."""
        for f in self.flows:
            yield from f.steps

    def all_target_names(self) -> Set[str]:
        """Collect all unique target names referenced in steps."""
        return {s.target_name for s in self.all_steps() if s.target_name}

    def all_target_roles(self) -> Set[str]:
        """Collect all unique target roles referenced in steps."""
        return {s.target_role for s in self.all_steps() if s.target_role}

    def get_critical_path(self) -> List[TaskStep]:
        """Return the longest required-step chain across all flows.

        The critical path is the flow with the most required (non-optional)
        steps, returning only those steps.
        """
        best: List[TaskStep] = []
        for f in self.flows:
            required = f.required_steps()
            if len(required) > len(best):
                best = required
        return best

    # -- validation ----------------------------------------------------------

    def validate(self) -> List[str]:
        """Run basic structural validation.  Return a list of error strings
        (empty if valid).
        """
        errors: List[str] = []
        if not self.name:
            errors.append("TaskSpec.name is empty.")
        if not self.flows:
            errors.append("TaskSpec has no flows.")

        seen_flow_ids: Set[str] = set()
        for f in self.flows:
            if f.flow_id in seen_flow_ids:
                errors.append(f"Duplicate flow_id: {f.flow_id}")
            seen_flow_ids.add(f.flow_id)

            if not f.steps:
                errors.append(f"Flow {f.flow_id!r} has no steps.")

            seen_step_ids: Set[str] = set()
            for s in f.steps:
                if s.step_id in seen_step_ids:
                    errors.append(
                        f"Duplicate step_id {s.step_id!r} in flow {f.flow_id!r}."
                    )
                seen_step_ids.add(s.step_id)

                if s.action_type not in ACTION_TYPES:
                    errors.append(
                        f"Step {s.step_id!r}: unknown action_type {s.action_type!r}."
                    )

                if s.is_input_action and not s.input_value and not s.optional:
                    errors.append(
                        f"Step {s.step_id!r}: input action with no input_value "
                        f"and not marked optional."
                    )

                for dep in s.depends_on:
                    if dep not in seen_step_ids:
                        errors.append(
                            f"Step {s.step_id!r}: dependency {dep!r} not found "
                            f"in preceding steps of flow {f.flow_id!r}."
                        )

        return errors

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "name": self.name,
            "description": self.description,
            "flows": [f.to_dict() for f in self.flows],
            "initial_state": copy.deepcopy(self.initial_state),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        return cls(
            spec_id=data.get("spec_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            flows=[TaskFlow.from_dict(f) for f in data.get("flows", [])],
            initial_state=dict(data.get("initial_state", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def deep_copy(self) -> "TaskSpec":
        return TaskSpec.from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# TaskGraph — dependency DAG over steps
# ---------------------------------------------------------------------------


@dataclass
class TaskGraph:
    """A directed acyclic graph (DAG) representing step dependencies.

    Nodes are :class:`TaskStep` instances; edges encode ``depends_on``
    relationships.  Provides topological sorting, critical-path analysis,
    and parallel-group detection.
    """

    nodes: Dict[str, TaskStep] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    _reverse_edges: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_flow(cls, flow: TaskFlow) -> "TaskGraph":
        """Build a :class:`TaskGraph` from a :class:`TaskFlow`.

        Explicit ``depends_on`` edges are honoured; if a step has no explicit
        dependencies, an implicit dependency on the previous step in list
        order is added.
        """
        g = cls()
        for step in flow.steps:
            g.add_node(step)
        for i, step in enumerate(flow.steps):
            if step.depends_on:
                for dep in step.depends_on:
                    if dep in g.nodes:
                        g.add_edge(dep, step.step_id)
            elif i > 0:
                g.add_edge(flow.steps[i - 1].step_id, step.step_id)
        return g

    @classmethod
    def from_spec(cls, spec: TaskSpec) -> "TaskGraph":
        """Build a unified graph from all flows in a :class:`TaskSpec`.

        Flows are treated as independent sub-graphs; there are no
        cross-flow edges.
        """
        g = cls()
        for flow in spec.flows:
            sub = cls.from_flow(flow)
            g.nodes.update(sub.nodes)
            for src, dsts in sub.edges.items():
                g.edges[src].extend(dsts)
            for dst, srcs in sub._reverse_edges.items():
                g._reverse_edges[dst].extend(srcs)
        return g

    def add_node(self, step: TaskStep) -> None:
        self.nodes[step.step_id] = step
        if step.step_id not in self.edges:
            self.edges[step.step_id] = []

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge ``from_id -> to_id``."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(
                f"Both {from_id!r} and {to_id!r} must be in the graph."
            )
        self.edges[from_id].append(to_id)
        self._reverse_edges[to_id].append(from_id)

    # -- queries -------------------------------------------------------------

    def roots(self) -> List[str]:
        """Return node IDs with no incoming edges (entry points)."""
        return [
            nid for nid in self.nodes if not self._reverse_edges.get(nid)
        ]

    def leaves(self) -> List[str]:
        """Return node IDs with no outgoing edges (terminal steps)."""
        return [nid for nid in self.nodes if not self.edges.get(nid)]

    def predecessors(self, node_id: str) -> List[str]:
        return list(self._reverse_edges.get(node_id, []))

    def successors(self, node_id: str) -> List[str]:
        return list(self.edges.get(node_id, []))

    # -- topological sort ----------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Return a topological ordering of node IDs (Kahn's algorithm).

        Raises :class:`ValueError` if the graph contains a cycle.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for src, dsts in self.edges.items():
            for dst in dsts:
                in_degree[dst] += 1

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dst in self.edges.get(nid, []):
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    queue.append(dst)

        if len(order) != len(self.nodes):
            raise ValueError("TaskGraph contains a cycle — topological sort impossible.")

        return order

    # -- critical path -------------------------------------------------------

    def critical_path(self, weights: Optional[Dict[str, float]] = None) -> List[str]:
        """Compute the critical (longest) path through the DAG.

        Parameters
        ----------
        weights : dict[str, float] | None
            Optional per-node weights.  Defaults to 1.0 per node.

        Returns
        -------
        list[str]
            Ordered list of node IDs on the critical path.
        """
        if weights is None:
            weights = {nid: 1.0 for nid in self.nodes}

        order = self.topological_sort()
        dist: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        pred: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}

        for nid in order:
            for dst in self.edges.get(nid, []):
                new_dist = dist[nid] + weights.get(dst, 1.0)
                if new_dist > dist[dst]:
                    dist[dst] = new_dist
                    pred[dst] = nid

        # backtrack from the heaviest node
        end = max(dist, key=lambda k: dist[k])
        path: List[str] = []
        cur: Optional[str] = end
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()
        return path

    # -- parallel groups -----------------------------------------------------

    def parallel_groups(self) -> List[Set[str]]:
        """Detect groups of steps that *can* execute concurrently.

        Two steps belong to the same parallel group if neither is an
        ancestor of the other in the DAG.  We approximate this by level
        sets (all nodes at the same depth in a topological layering).
        """
        order = self.topological_sort()
        level: Dict[str, int] = {}
        for nid in order:
            preds = self._reverse_edges.get(nid, [])
            if not preds:
                level[nid] = 0
            else:
                level[nid] = max(level[p] for p in preds) + 1

        groups_map: Dict[int, Set[str]] = defaultdict(set)
        for nid, lev in level.items():
            groups_map[lev].add(nid)

        return [groups_map[k] for k in sorted(groups_map)]

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: s.to_dict() for nid, s in self.nodes.items()},
            "edges": {src: list(dsts) for src, dsts in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        g = cls()
        for nid, sdata in data.get("nodes", {}).items():
            g.nodes[nid] = TaskStep.from_dict(sdata)
        for src, dsts in data.get("edges", {}).items():
            g.edges[src] = list(dsts)
            for dst in dsts:
                g._reverse_edges[dst].append(src)
        return g

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, step_id: str) -> bool:
        return step_id in self.nodes

    def __iter__(self) -> Iterator[str]:
        return iter(self.topological_sort())
