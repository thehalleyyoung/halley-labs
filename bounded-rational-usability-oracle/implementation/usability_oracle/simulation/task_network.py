"""
usability_oracle.simulation.task_network — Task network representation.

Models task decomposition as a directed acyclic graph (DAG) for
scheduling, critical path analysis, and resource leveling.  Supports
CPM (Critical Path Method) and PERT (Program Evaluation and Review
Technique) analysis.

The task network complements GOMS by providing a more formal graph-based
representation that enables parallel execution analysis, resource
contention detection, and what-if design exploration.

References:
    Kelley, J. E., & Walker, M. R. (1959). Critical-path planning and
        scheduling. *Proceedings of the Eastern Joint Computer Conference*,
        160-173.
    Malcolm, D. G. et al. (1959). Application of a technique for
        research and development program evaluation. *Operations Research*,
        7(5), 646-669.
    John, B. E. (1990). Extensions of GOMS analyses to expert
        performance. *Proceedings of CHI '90*, 107-115.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Task status
# ═══════════════════════════════════════════════════════════════════════════

@unique
class TaskStatus(Enum):
    """Execution status of a task node."""
    NOT_STARTED = auto()
    READY = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()
    SKIPPED = auto()


# ═══════════════════════════════════════════════════════════════════════════
# Resource model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Resource:
    """A cognitive or physical resource with limited capacity.

    Attributes:
        name: Resource identifier (e.g., 'right_hand', 'visual_attention').
        capacity: Maximum concurrent utilization (default 1 for single-resource).
        current_load: Current utilization count.
    """
    name: str = ""
    capacity: int = 1
    current_load: int = 0

    @property
    def available(self) -> int:
        return max(self.capacity - self.current_load, 0)

    @property
    def is_available(self) -> bool:
        return self.current_load < self.capacity

    def allocate(self) -> bool:
        if self.current_load >= self.capacity:
            return False
        self.current_load += 1
        return True

    def release(self) -> None:
        self.current_load = max(0, self.current_load - 1)

    def reset(self) -> None:
        self.current_load = 0


# ═══════════════════════════════════════════════════════════════════════════
# TaskNode
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskNode:
    """A node in the task decomposition network.

    Attributes:
        task_id: Unique identifier.
        name: Human-readable name.
        duration: Expected duration (seconds).
        duration_optimistic: Optimistic duration for PERT (seconds).
        duration_pessimistic: Pessimistic duration for PERT (seconds).
        resource_requirements: Set of resource names needed during execution.
        predecessors: Task IDs that must complete before this task starts.
        successors: Task IDs that depend on this task.
        status: Current execution status.
        earliest_start: Earliest possible start time (computed by forward pass).
        earliest_finish: Earliest possible finish time.
        latest_start: Latest allowable start time (computed by backward pass).
        latest_finish: Latest allowable finish time.
        total_float: Schedule flexibility (LS - ES).
        free_float: Free float (earliest successor ES - EF).
        actual_start: Actual start time (for simulation tracking).
        actual_finish: Actual finish time.
        metadata: Additional properties.
    """
    task_id: str = ""
    name: str = ""
    duration: float = 0.0
    duration_optimistic: float = 0.0
    duration_pessimistic: float = 0.0
    resource_requirements: Set[str] = field(default_factory=set)
    predecessors: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.NOT_STARTED
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    latest_start: float = float("inf")
    latest_finish: float = float("inf")
    total_float: float = float("inf")
    free_float: float = float("inf")
    actual_start: Optional[float] = None
    actual_finish: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pert_expected_duration(self) -> float:
        """PERT expected duration: (O + 4M + P) / 6.

        Reference: Malcolm et al. (1959).
        """
        o = self.duration_optimistic if self.duration_optimistic > 0 else self.duration * 0.8
        p = self.duration_pessimistic if self.duration_pessimistic > 0 else self.duration * 1.4
        return (o + 4 * self.duration + p) / 6.0

    @property
    def pert_variance(self) -> float:
        """PERT variance: ((P - O) / 6)^2.

        Reference: Malcolm et al. (1959).
        """
        o = self.duration_optimistic if self.duration_optimistic > 0 else self.duration * 0.8
        p = self.duration_pessimistic if self.duration_pessimistic > 0 else self.duration * 1.4
        return ((p - o) / 6.0) ** 2

    @property
    def is_critical(self) -> bool:
        """A task is critical if its total float is zero (or near-zero)."""
        return abs(self.total_float) < 1e-9

    @property
    def is_ready(self) -> bool:
        return self.status in (TaskStatus.NOT_STARTED, TaskStatus.READY)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "duration": self.duration,
            "predecessors": list(self.predecessors),
            "successors": list(self.successors),
            "status": self.status.name,
            "earliest_start": self.earliest_start,
            "earliest_finish": self.earliest_finish,
            "latest_start": self.latest_start,
            "latest_finish": self.latest_finish,
            "total_float": self.total_float,
            "is_critical": self.is_critical,
        }


# ═══════════════════════════════════════════════════════════════════════════
# TaskNetwork
# ═══════════════════════════════════════════════════════════════════════════

class TaskNetwork:
    """Directed acyclic graph of tasks for scheduling and analysis.

    The network supports:
    - Topological ordering
    - Critical path analysis (CPM/PERT)
    - Resource-constrained scheduling
    - Parallel decomposition
    - What-if analysis

    Usage::

        net = TaskNetwork()
        net.add_task(TaskNode(task_id="A", name="Click button", duration=1.1))
        net.add_task(TaskNode(task_id="B", name="Type text", duration=2.0))
        net.add_dependency("B", "A")
        net.compute_schedule()
        cp = net.critical_path_analysis()
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, TaskNode] = {}
        self._resources: Dict[str, Resource] = {}
        self._computed: bool = False

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def add_task(self, node: TaskNode) -> None:
        """Add a task node to the network."""
        self._nodes[node.task_id] = node
        self._computed = False

    def add_tasks(self, nodes: List[TaskNode]) -> None:
        """Add multiple task nodes."""
        for node in nodes:
            self.add_task(node)

    def add_dependency(self, task_id: str, depends_on: str) -> bool:
        """Add a dependency: task_id depends on depends_on.

        Returns False if either task doesn't exist or would create a cycle.
        """
        if task_id not in self._nodes or depends_on not in self._nodes:
            return False
        # Cycle check
        if self._would_create_cycle(task_id, depends_on):
            return False
        self._nodes[task_id].predecessors.add(depends_on)
        self._nodes[depends_on].successors.add(task_id)
        self._computed = False
        return True

    def add_resource(self, resource: Resource) -> None:
        """Register a resource for resource-constrained scheduling."""
        self._resources[resource.name] = resource

    def remove_task(self, task_id: str) -> bool:
        """Remove a task and all its dependencies."""
        if task_id not in self._nodes:
            return False
        node = self._nodes[task_id]
        for pred_id in node.predecessors:
            if pred_id in self._nodes:
                self._nodes[pred_id].successors.discard(task_id)
        for succ_id in node.successors:
            if succ_id in self._nodes:
                self._nodes[succ_id].predecessors.discard(task_id)
        del self._nodes[task_id]
        self._computed = False
        return True

    def _would_create_cycle(self, task_id: str, new_predecessor: str) -> bool:
        """Check if adding new_predecessor → task_id creates a cycle."""
        visited: Set[str] = set()
        queue = deque([task_id])
        while queue:
            current = queue.popleft()
            if current == new_predecessor:
                return True
            if current in visited:
                continue
            visited.add(current)
            node = self._nodes.get(current)
            if node:
                for succ in node.successors:
                    queue.append(succ)
        return False

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Return task IDs in topological order (Kahn's algorithm).

        Tasks with no predecessors come first.  Ties broken by task_id
        for determinism.

        Returns:
            List of task IDs in execution order.

        Raises:
            ValueError: If the network contains a cycle.
        """
        in_degree: Dict[str, int] = {tid: len(n.predecessors) for tid, n in self._nodes.items()}
        queue: List[str] = sorted([tid for tid, d in in_degree.items() if d == 0])
        result: List[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            node = self._nodes[current]
            for succ_id in sorted(node.successors):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
                    queue.sort()

        if len(result) != len(self._nodes):
            raise ValueError("Task network contains a cycle")

        return result

    # ------------------------------------------------------------------
    # CPM / PERT scheduling
    # ------------------------------------------------------------------

    def compute_schedule(self) -> None:
        """Compute earliest/latest times using forward and backward passes.

        This implements the Critical Path Method (CPM):
        - Forward pass: compute ES and EF for each task.
        - Backward pass: compute LS and LF for each task.
        - Float computation: TF = LS - ES.

        Reference: Kelley & Walker (1959).
        """
        order = self.topological_sort()

        # Forward pass
        for tid in order:
            node = self._nodes[tid]
            if not node.predecessors:
                node.earliest_start = 0.0
            else:
                node.earliest_start = max(
                    self._nodes[p].earliest_finish
                    for p in node.predecessors
                    if p in self._nodes
                )
            node.earliest_finish = node.earliest_start + node.duration

        # Project completion time
        project_end = max(n.earliest_finish for n in self._nodes.values()) if self._nodes else 0.0

        # Backward pass
        for tid in reversed(order):
            node = self._nodes[tid]
            if not node.successors:
                node.latest_finish = project_end
            else:
                node.latest_finish = min(
                    self._nodes[s].latest_start
                    for s in node.successors
                    if s in self._nodes
                )
            node.latest_start = node.latest_finish - node.duration

        # Float computation
        for node in self._nodes.values():
            node.total_float = node.latest_start - node.earliest_start

            # Free float
            if node.successors:
                min_succ_es = min(
                    self._nodes[s].earliest_start
                    for s in node.successors
                    if s in self._nodes
                )
                node.free_float = min_succ_es - node.earliest_finish
            else:
                node.free_float = project_end - node.earliest_finish

        self._computed = True

    def critical_path_analysis(self) -> Dict[str, Any]:
        """Perform critical path analysis.

        Returns:
            Dict with critical path tasks, total time, variance, and probability.
        """
        if not self._computed:
            self.compute_schedule()

        critical_tasks = [
            node for node in self._nodes.values()
            if node.is_critical
        ]
        critical_ids = [n.task_id for n in critical_tasks]

        # Build ordered critical path
        order = self.topological_sort()
        critical_order = [tid for tid in order if tid in set(critical_ids)]

        total_time = sum(n.duration for n in critical_tasks)
        total_variance = sum(n.pert_variance for n in critical_tasks)
        total_std = math.sqrt(total_variance) if total_variance > 0 else 0.0

        project_end = max(
            (n.earliest_finish for n in self._nodes.values()), default=0.0
        )

        return {
            "critical_path": critical_order,
            "critical_tasks": [n.to_dict() for n in critical_tasks],
            "project_duration": project_end,
            "total_critical_time": total_time,
            "total_variance": total_variance,
            "total_std": total_std,
            "n_critical_tasks": len(critical_tasks),
            "n_total_tasks": len(self._nodes),
            "critical_ratio": len(critical_tasks) / max(len(self._nodes), 1),
        }

    # ------------------------------------------------------------------
    # Parallel decomposition
    # ------------------------------------------------------------------

    def parallel_decomposition(self) -> List[List[str]]:
        """Identify groups of tasks that can execute in parallel.

        Tasks in the same group have no dependency relationship and can
        theoretically execute simultaneously if resources allow.

        Returns:
            List of task-ID groups, ordered by earliest start time.
        """
        if not self._computed:
            self.compute_schedule()

        # Group by earliest start time (with tolerance)
        groups: Dict[float, List[str]] = defaultdict(list)
        tolerance = 1e-6
        for tid, node in self._nodes.items():
            rounded_es = round(node.earliest_start / tolerance) * tolerance
            groups[rounded_es].append(tid)

        # Sort groups by time
        sorted_times = sorted(groups.keys())
        return [sorted(groups[t]) for t in sorted_times]

    def max_parallelism(self) -> int:
        """Return the maximum number of tasks active simultaneously."""
        groups = self.parallel_decomposition()
        return max(len(g) for g in groups) if groups else 0

    # ------------------------------------------------------------------
    # Resource leveling
    # ------------------------------------------------------------------

    def resource_leveling(
        self,
        available_resources: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Schedule tasks respecting resource constraints.

        Uses a simple priority-based heuristic: tasks on the critical path
        are scheduled first; ties broken by earliest start time.

        Parameters:
            available_resources: Dict of resource_name -> capacity.
                If None, uses registered resources.

        Returns:
            Dict with scheduled start/finish times and resource utilization.
        """
        if not self._computed:
            self.compute_schedule()

        resources: Dict[str, int] = {}
        if available_resources:
            resources = dict(available_resources)
        else:
            resources = {r.name: r.capacity for r in self._resources.values()}

        # If no resources specified, assume single unit for each requirement
        all_requirements: Set[str] = set()
        for node in self._nodes.values():
            all_requirements.update(node.resource_requirements)
        for req in all_requirements:
            if req not in resources:
                resources[req] = 1

        order = self.topological_sort()
        schedule: Dict[str, Tuple[float, float]] = {}
        resource_timeline: Dict[str, List[Tuple[float, float]]] = {r: [] for r in resources}

        # Priority: critical tasks first, then by ES
        priority_order = sorted(order, key=lambda t: (
            0 if self._nodes[t].is_critical else 1,
            self._nodes[t].earliest_start,
            t,
        ))

        for tid in priority_order:
            node = self._nodes[tid]
            # Earliest possible from predecessors
            pred_finish = 0.0
            for p in node.predecessors:
                if p in schedule:
                    pred_finish = max(pred_finish, schedule[p][1])

            start = max(node.earliest_start, pred_finish)

            # Check resource availability
            if node.resource_requirements:
                for res_name in node.resource_requirements:
                    if res_name in resource_timeline:
                        for alloc_start, alloc_end in resource_timeline[res_name]:
                            if start < alloc_end and (start + node.duration) > alloc_start:
                                start = max(start, alloc_end)

            finish = start + node.duration
            schedule[tid] = (start, finish)

            # Record resource allocation
            for res_name in node.resource_requirements:
                if res_name in resource_timeline:
                    resource_timeline[res_name].append((start, finish))

        # Compute utilizations
        project_end = max(f for _, f in schedule.values()) if schedule else 0.0
        utilizations: Dict[str, float] = {}
        for res_name, allocations in resource_timeline.items():
            total_busy = sum(end - start for start, end in allocations)
            cap = resources.get(res_name, 1)
            utilizations[res_name] = total_busy / (project_end * cap) if project_end > 0 and cap > 0 else 0.0

        return {
            "schedule": {tid: {"start": s, "finish": f} for tid, (s, f) in schedule.items()},
            "project_duration": project_end,
            "resource_utilizations": utilizations,
            "resource_timeline": {r: [(s, f) for s, f in allocs] for r, allocs in resource_timeline.items()},
        }

    # ------------------------------------------------------------------
    # What-if analysis
    # ------------------------------------------------------------------

    def what_if_analysis(
        self,
        removed_tasks: Optional[List[str]] = None,
        modified_durations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Analyze impact of removing or changing tasks.

        Parameters:
            removed_tasks: Task IDs to remove from the network.
            modified_durations: Dict of task_id -> new_duration.

        Returns:
            Dict comparing original vs modified network.
        """
        # Original analysis
        self.compute_schedule()
        original_cpa = self.critical_path_analysis()
        original_duration = original_cpa["project_duration"]

        # Create modified network
        modified = TaskNetwork()
        for tid, node in self._nodes.items():
            if removed_tasks and tid in removed_tasks:
                continue
            new_node = TaskNode(
                task_id=node.task_id,
                name=node.name,
                duration=node.duration,
                duration_optimistic=node.duration_optimistic,
                duration_pessimistic=node.duration_pessimistic,
                resource_requirements=set(node.resource_requirements),
                predecessors=set(node.predecessors),
                successors=set(node.successors),
            )
            if modified_durations and tid in modified_durations:
                new_node.duration = modified_durations[tid]
            # Remove references to removed tasks
            if removed_tasks:
                new_node.predecessors -= set(removed_tasks)
                new_node.successors -= set(removed_tasks)
            modified.add_task(new_node)

        try:
            modified.compute_schedule()
            modified_cpa = modified.critical_path_analysis()
            modified_duration = modified_cpa["project_duration"]
        except ValueError:
            modified_cpa = {}
            modified_duration = 0.0

        delta = modified_duration - original_duration
        pct_change = (delta / original_duration * 100.0) if original_duration > 0 else 0.0

        return {
            "original_duration": original_duration,
            "modified_duration": modified_duration,
            "delta_seconds": delta,
            "pct_change": pct_change,
            "is_improvement": delta < 0,
            "original_critical_path": original_cpa.get("critical_path", []),
            "modified_critical_path": modified_cpa.get("critical_path", []),
            "removed_tasks": removed_tasks or [],
            "modified_durations": modified_durations or {},
        }

    # ------------------------------------------------------------------
    # Visualization support
    # ------------------------------------------------------------------

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Export the network as an adjacency matrix.

        Returns:
            Tuple of (matrix, task_id_list) where matrix[i][j] = 1 means
            task i is a predecessor of task j.
        """
        ids = sorted(self._nodes.keys())
        id_to_idx = {tid: i for i, tid in enumerate(ids)}
        n = len(ids)
        matrix = np.zeros((n, n), dtype=int)

        for tid, node in self._nodes.items():
            i = id_to_idx[tid]
            for succ in node.successors:
                if succ in id_to_idx:
                    j = id_to_idx[succ]
                    matrix[i][j] = 1

        return matrix, ids

    def to_dot(self) -> str:
        """Export as Graphviz DOT format for visualization."""
        lines = ["digraph TaskNetwork {", "  rankdir=LR;"]
        for tid, node in self._nodes.items():
            label = f"{node.name}\\n{node.duration:.2f}s"
            style = "filled" if node.is_critical else ""
            color = "salmon" if node.is_critical else "lightblue"
            lines.append(f'  "{tid}" [label="{label}", style="{style}", fillcolor="{color}"];')
        for tid, node in self._nodes.items():
            for succ in sorted(node.successors):
                lines.append(f'  "{tid}" -> "{succ}";')
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Network queries
    # ------------------------------------------------------------------

    @property
    def n_tasks(self) -> int:
        return len(self._nodes)

    @property
    def n_dependencies(self) -> int:
        return sum(len(n.predecessors) for n in self._nodes.values())

    @property
    def task_ids(self) -> List[str]:
        return sorted(self._nodes.keys())

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self._nodes.get(task_id)

    def source_tasks(self) -> List[str]:
        """Tasks with no predecessors (entry points)."""
        return [tid for tid, n in self._nodes.items() if not n.predecessors]

    def sink_tasks(self) -> List[str]:
        """Tasks with no successors (exit points)."""
        return [tid for tid, n in self._nodes.items() if not n.successors]

    def longest_path_length(self) -> float:
        """The project makespan (longest path through the network)."""
        if not self._computed:
            self.compute_schedule()
        return max((n.earliest_finish for n in self._nodes.values()), default=0.0)

    def all_paths(self) -> List[List[str]]:
        """Enumerate all source-to-sink paths (exponential in general)."""
        sources = self.source_tasks()
        sinks = set(self.sink_tasks())
        paths: List[List[str]] = []

        def _dfs(current: str, path: List[str]) -> None:
            if current in sinks:
                paths.append(list(path))
                # Don't return here — a sink might also be intermediate
                if not self._nodes[current].successors:
                    return
            for succ in sorted(self._nodes[current].successors):
                path.append(succ)
                _dfs(succ, path)
                path.pop()

        for src in sources:
            _dfs(src, [src])
        return paths

    def summary(self) -> str:
        """Human-readable network summary."""
        if not self._computed:
            self.compute_schedule()
        cpa = self.critical_path_analysis()
        lines = [
            f"Task Network Summary:",
            f"  Tasks: {self.n_tasks}",
            f"  Dependencies: {self.n_dependencies}",
            f"  Sources: {len(self.source_tasks())}",
            f"  Sinks: {len(self.sink_tasks())}",
            f"  Project duration: {cpa['project_duration']:.3f}s",
            f"  Critical tasks: {cpa['n_critical_tasks']} ({cpa['critical_ratio']:.1%})",
            f"  Max parallelism: {self.max_parallelism()}",
        ]
        return "\n".join(lines)
