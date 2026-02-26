"""
Execution trace management for model checking.

Provides recording, construction, minimization, serialization, and
pretty-printing of execution traces and lasso counterexamples.
"""

from __future__ import annotations

import json
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .graph import TransitionGraph, StateNode, TransitionEdge


# ---------------------------------------------------------------------------
# Trace step
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """One step in a trace: state + outgoing action."""

    state_hash: str
    state: Dict[str, Any]
    action_label: Optional[str] = None
    depth: int = 0
    atomic_propositions: Optional[Set[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "state": self.state,
            "action_label": self.action_label,
            "depth": self.depth,
            "atomic_propositions": (
                sorted(self.atomic_propositions) if self.atomic_propositions else []
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceStep":
        return cls(
            state_hash=d["state_hash"],
            state=d["state"],
            action_label=d.get("action_label"),
            depth=d.get("depth", 0),
            atomic_propositions=(
                set(d["atomic_propositions"]) if d.get("atomic_propositions") else None
            ),
        )


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

class ExecutionTrace:
    """
    A finite sequence of (state, action) pairs representing a system run.
    """

    def __init__(self, steps: Optional[List[TraceStep]] = None) -> None:
        self._steps: List[TraceStep] = list(steps) if steps else []

    # -- Construction ------------------------------------------------------

    def append(self, step: TraceStep) -> None:
        self._steps.append(step)

    def extend(self, steps: List[TraceStep]) -> None:
        self._steps.extend(steps)

    @classmethod
    def from_state_sequence(
        cls,
        graph: TransitionGraph,
        state_hashes: List[str],
    ) -> "ExecutionTrace":
        """Build a trace from a list of state hashes using graph data."""
        trace = cls()
        for i, h in enumerate(state_hashes):
            node = graph.get_state(h)
            if node is None:
                continue
            action: Optional[str] = None
            if i < len(state_hashes) - 1:
                next_h = state_hashes[i + 1]
                edges = graph.get_edges(h, next_h)
                if edges:
                    action = edges[0].action_label
            trace.append(
                TraceStep(
                    state_hash=h,
                    state=node.full_state,
                    action_label=action,
                    depth=i,
                    atomic_propositions=set(node.atomic_propositions),
                )
            )
        return trace

    @classmethod
    def from_path(
        cls,
        graph: TransitionGraph,
        path: List[str],
    ) -> "ExecutionTrace":
        return cls.from_state_sequence(graph, path)

    # -- Access ------------------------------------------------------------

    @property
    def steps(self) -> List[TraceStep]:
        return list(self._steps)

    @property
    def length(self) -> int:
        return len(self._steps)

    @property
    def is_empty(self) -> bool:
        return len(self._steps) == 0

    def first_state(self) -> Optional[TraceStep]:
        return self._steps[0] if self._steps else None

    def last_state(self) -> Optional[TraceStep]:
        return self._steps[-1] if self._steps else None

    def state_at(self, index: int) -> Optional[TraceStep]:
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    def state_hashes(self) -> List[str]:
        return [s.state_hash for s in self._steps]

    def actions(self) -> List[Optional[str]]:
        return [s.action_label for s in self._steps]

    def contains_state(self, state_hash: str) -> bool:
        return any(s.state_hash == state_hash for s in self._steps)

    # -- Validation --------------------------------------------------------

    def is_valid(self, graph: TransitionGraph) -> bool:
        """Check that consecutive states are connected by transitions."""
        for i in range(len(self._steps) - 1):
            src = self._steps[i].state_hash
            tgt = self._steps[i + 1].state_hash
            if not graph.get_edges(src, tgt):
                return False
        return True

    def starts_from_initial(self, graph: TransitionGraph) -> bool:
        if not self._steps:
            return True
        return self._steps[0].state_hash in graph.initial_states

    # -- Serialization -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {"steps": [s.to_dict() for s in self._steps]},
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionTrace":
        data = json.loads(json_str)
        steps = [TraceStep.from_dict(d) for d in data.get("steps", [])]
        return cls(steps)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ExecutionTrace":
        with open(path) as f:
            return cls.from_json(f.read())

    # -- Pretty printing ---------------------------------------------------

    def pretty_print(
        self,
        *,
        show_full_state: bool = True,
        show_propositions: bool = True,
        max_width: int = 80,
        var_filter: Optional[Set[str]] = None,
    ) -> str:
        lines: List[str] = []
        lines.append(f"=== Execution Trace ({self.length} steps) ===")
        lines.append("")

        for i, step in enumerate(self._steps):
            lines.append(f"--- State {i} (depth {step.depth}) ---")
            lines.append(f"  Hash: {step.state_hash[:16]}...")

            if show_full_state:
                state = step.state
                if var_filter:
                    state = {k: v for k, v in state.items() if k in var_filter}
                for var, val in sorted(state.items()):
                    val_str = json.dumps(val, default=str)
                    if len(val_str) > max_width - 10:
                        val_str = val_str[: max_width - 13] + "..."
                    lines.append(f"  {var} = {val_str}")

            if show_propositions and step.atomic_propositions:
                props = ", ".join(sorted(step.atomic_propositions))
                lines.append(f"  Props: {{{props}}}")

            if step.action_label:
                lines.append(f"  --[ {step.action_label} ]-->")
            lines.append("")

        lines.append("=== End of Trace ===")
        return "\n".join(lines)

    def diff_print(self, max_width: int = 80) -> str:
        """Print only variables that change between consecutive states."""
        lines: List[str] = []
        lines.append(f"=== Trace Diff ({self.length} steps) ===")
        lines.append("")

        prev_state: Optional[Dict[str, Any]] = None
        for i, step in enumerate(self._steps):
            lines.append(f"--- State {i} ---")
            if prev_state is None:
                for var, val in sorted(step.state.items()):
                    val_str = json.dumps(val, default=str)
                    lines.append(f"  {var} = {val_str}")
            else:
                changed = False
                for var in sorted(set(step.state.keys()) | set(prev_state.keys())):
                    old_val = prev_state.get(var)
                    new_val = step.state.get(var)
                    if old_val != new_val:
                        changed = True
                        old_str = json.dumps(old_val, default=str) if old_val is not None else "<undef>"
                        new_str = json.dumps(new_val, default=str) if new_val is not None else "<undef>"
                        lines.append(f"  {var}: {old_str} -> {new_str}")
                if not changed:
                    lines.append("  (no change — stuttering)")

            if step.action_label:
                lines.append(f"  --[ {step.action_label} ]-->")
            lines.append("")
            prev_state = step.state

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ExecutionTrace(length={self.length})"


# ---------------------------------------------------------------------------
# Lasso trace (for liveness counterexamples)
# ---------------------------------------------------------------------------

class LassoTrace:
    """
    A lasso-shaped trace: finite prefix followed by an infinite loop.

    Used as counterexamples for liveness properties: the system reaches
    the loop entry and then cycles forever, violating the property.
    """

    def __init__(
        self,
        prefix: ExecutionTrace,
        loop: ExecutionTrace,
    ) -> None:
        self._prefix = prefix
        self._loop = loop

    @property
    def prefix(self) -> ExecutionTrace:
        return self._prefix

    @property
    def loop(self) -> ExecutionTrace:
        return self._loop

    @property
    def prefix_length(self) -> int:
        return self._prefix.length

    @property
    def loop_length(self) -> int:
        return self._loop.length

    @property
    def total_length(self) -> int:
        return self._prefix.length + self._loop.length

    @classmethod
    def from_state_lists(
        cls,
        graph: TransitionGraph,
        prefix_hashes: List[str],
        loop_hashes: List[str],
    ) -> "LassoTrace":
        prefix = ExecutionTrace.from_state_sequence(graph, prefix_hashes)
        loop = ExecutionTrace.from_state_sequence(graph, loop_hashes)
        return cls(prefix, loop)

    def is_valid(self, graph: TransitionGraph) -> bool:
        if not self._prefix.is_valid(graph):
            return False
        if not self._loop.is_valid(graph):
            return False
        if self._prefix.length > 0 and self._loop.length > 0:
            last_prefix = self._prefix.last_state()
            first_loop = self._loop.first_state()
            if last_prefix and first_loop:
                if not graph.get_edges(last_prefix.state_hash, first_loop.state_hash):
                    return False
        if self._loop.length > 0:
            last_loop = self._loop.last_state()
            first_loop = self._loop.first_state()
            if last_loop and first_loop:
                if not graph.get_edges(last_loop.state_hash, first_loop.state_hash):
                    return False
        return True

    def to_json(self) -> str:
        return json.dumps(
            {
                "prefix": json.loads(self._prefix.to_json()),
                "loop": json.loads(self._loop.to_json()),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LassoTrace":
        data = json.loads(json_str)
        prefix = ExecutionTrace.from_json(json.dumps(data["prefix"]))
        loop = ExecutionTrace.from_json(json.dumps(data["loop"]))
        return cls(prefix, loop)

    def pretty_print(self, **kwargs: Any) -> str:
        lines: List[str] = []
        lines.append("=== Lasso Counterexample ===")
        lines.append("")
        lines.append("--- PREFIX ---")
        lines.append(self._prefix.pretty_print(**kwargs))
        lines.append("")
        lines.append("--- LOOP (repeats forever) ---")
        lines.append(self._loop.pretty_print(**kwargs))
        lines.append("")
        lines.append("=== End of Lasso ===")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LassoTrace(prefix={self.prefix_length}, "
            f"loop={self.loop_length})"
        )


# ---------------------------------------------------------------------------
# Trace manager
# ---------------------------------------------------------------------------

class TraceManager:
    """
    Manages trace recording and counterexample construction over a
    transition graph.
    """

    def __init__(self, graph: TransitionGraph) -> None:
        self._graph = graph
        self._parent: Dict[str, Tuple[str, str]] = {}  # child -> (parent, action)
        self._recorded_traces: List[ExecutionTrace] = []

    # -- Recording ---------------------------------------------------------

    def record_transition(
        self, source_hash: str, target_hash: str, action: str
    ) -> None:
        """Record the parent of a state during exploration."""
        if target_hash not in self._parent:
            self._parent[target_hash] = (source_hash, action)

    def record_trace(self, trace: ExecutionTrace) -> None:
        self._recorded_traces.append(trace)

    @property
    def recorded_traces(self) -> List[ExecutionTrace]:
        return list(self._recorded_traces)

    # -- Counterexample construction ---------------------------------------

    def construct_trace_to(self, target_hash: str) -> Optional[ExecutionTrace]:
        """
        Reconstruct the shortest trace from an initial state to *target_hash*
        using BFS on the graph.
        """
        initials = self._graph.initial_states
        if target_hash in initials:
            node = self._graph.get_state(target_hash)
            if node:
                step = TraceStep(
                    state_hash=target_hash,
                    state=node.full_state,
                    depth=0,
                    atomic_propositions=set(node.atomic_propositions),
                )
                return ExecutionTrace([step])
            return None

        for init_h in initials:
            path = self._graph.shortest_path(init_h, target_hash)
            if path:
                return ExecutionTrace.from_state_sequence(self._graph, path)
        return None

    def construct_trace_from_parents(self, target_hash: str) -> Optional[ExecutionTrace]:
        """Reconstruct a trace using recorded parent links."""
        path_hashes: List[str] = [target_hash]
        current = target_hash
        visited: Set[str] = {current}

        while current in self._parent:
            parent_hash, action = self._parent[current]
            if parent_hash in visited:
                break
            visited.add(parent_hash)
            path_hashes.append(parent_hash)
            current = parent_hash

        path_hashes.reverse()
        return ExecutionTrace.from_state_sequence(self._graph, path_hashes)

    def construct_lasso(
        self,
        prefix_target: str,
        loop_back_to: str,
    ) -> Optional[LassoTrace]:
        """
        Construct a lasso counterexample: prefix to *prefix_target*,
        then a loop from *prefix_target* back to *loop_back_to* and
        from *loop_back_to* back to *prefix_target*.
        """
        prefix_trace = self.construct_trace_to(loop_back_to)
        if prefix_trace is None:
            return None

        loop_path = self._graph.shortest_path(loop_back_to, prefix_target)
        if loop_path is None:
            return None

        back_path = self._graph.shortest_path(prefix_target, loop_back_to)
        if back_path is None:
            loop_hashes = loop_path
        else:
            loop_hashes = loop_path + back_path[1:]

        loop_trace = ExecutionTrace.from_state_sequence(self._graph, loop_hashes)
        return LassoTrace(prefix_trace, loop_trace)

    # -- Trace minimization ------------------------------------------------

    def minimize_trace(self, trace: ExecutionTrace) -> ExecutionTrace:
        """
        Find the shortest trace from the same initial state to the same
        error state by using BFS over the graph.
        """
        if trace.length <= 1:
            return trace

        start = trace.first_state()
        end = trace.last_state()
        if start is None or end is None:
            return trace

        path = self._graph.shortest_path(start.state_hash, end.state_hash)
        if path is None:
            return trace

        return ExecutionTrace.from_state_sequence(self._graph, path)

    def minimize_lasso(self, lasso: LassoTrace) -> LassoTrace:
        """Minimize both prefix and loop of a lasso trace."""
        min_prefix = self.minimize_trace(lasso.prefix)
        min_loop = self.minimize_trace(lasso.loop)
        return LassoTrace(min_prefix, min_loop)

    # -- Filtering ---------------------------------------------------------

    def filter_traces_by_action(
        self, action: str
    ) -> List[ExecutionTrace]:
        return [
            t
            for t in self._recorded_traces
            if any(s.action_label == action for s in t.steps)
        ]

    def filter_traces_containing_state(
        self, state_hash: str
    ) -> List[ExecutionTrace]:
        return [
            t for t in self._recorded_traces if t.contains_state(state_hash)
        ]

    # -- Statistics --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        lengths = [t.length for t in self._recorded_traces]
        return {
            "num_traces": len(self._recorded_traces),
            "parent_links": len(self._parent),
            "avg_trace_length": (
                sum(lengths) / len(lengths) if lengths else 0
            ),
            "max_trace_length": max(lengths, default=0),
            "min_trace_length": min(lengths, default=0),
        }
