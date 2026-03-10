"""
usability_oracle.goms.critical_path — Critical path analysis for GOMS models.

Computes the critical path through the GOMS operator network, identifies
bottleneck operators, computes slack times, performs what-if analysis for
operator time changes, and generates Gantt-chart-like schedules for
parallel operations.

References
----------
John, B. E. & Kieras, D. E. (1996). The GOMS family of user interface
analysis techniques: comparison and contrast. *ACM Transactions on
Computer-Human Interaction*, 3(4), 320-351.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from usability_oracle.core.constants import (
    WORKING_MEMORY_CAPACITY,
)
from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    GomsTrace,
    OperatorType,
)


# ═══════════════════════════════════════════════════════════════════════════
# Operator node (for DAG construction)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OperatorNode:
    """A node in the operator DAG with scheduling information."""

    index: int
    operator: GomsOperator
    method_id: str
    goal_id: str
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    # Forward pass
    earliest_start: float = 0.0
    earliest_finish: float = 0.0
    # Backward pass
    latest_start: float = float("inf")
    latest_finish: float = float("inf")

    @property
    def slack(self) -> float:
        """Total slack (float time) for this operator."""
        return self.latest_start - self.earliest_start

    @property
    def is_critical(self) -> bool:
        """True if this operator is on the critical path (zero slack)."""
        return abs(self.slack) < 1e-9

    @property
    def duration_s(self) -> float:
        return self.operator.duration_s

    @property
    def channel(self) -> str:
        """Cognitive processing channel for this operator."""
        if self.operator.is_cognitive:
            return "cognitive"
        if self.operator.is_motor:
            return "motor"
        if self.operator.op_type.is_system:
            return "system"
        return "other"


# ═══════════════════════════════════════════════════════════════════════════
# Schedule entry (Gantt chart)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScheduleEntry:
    """One bar in a Gantt-chart schedule."""

    operator_index: int
    operator: GomsOperator
    method_id: str
    channel: str
    start_s: float
    end_s: float
    is_critical: bool
    slack_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck classification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Bottleneck:
    """An identified bottleneck in the GOMS operator network."""

    operator_index: int
    operator: GomsOperator
    bottleneck_type: str
    """One of: critical_path, resource_contention, memory_overload, motor_difficulty."""
    severity: float
    """0-1 severity score."""
    description: str


# ═══════════════════════════════════════════════════════════════════════════
# CriticalPathAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class CriticalPathAnalyzer:
    """Analyse the critical path through a GOMS operator network.

    Builds a DAG of operators, computes forward/backward passes,
    identifies the critical path, computes slack times, and provides
    what-if analysis capabilities.
    """

    def __init__(self) -> None:
        self._nodes: List[OperatorNode] = []
        self._makespan: float = 0.0
        self._built = False

    @property
    def nodes(self) -> Sequence[OperatorNode]:
        return self._nodes

    @property
    def makespan(self) -> float:
        """Total project duration (critical path length)."""
        return self._makespan

    # ── DAG construction ──────────────────────────────────────────────

    def build_dag(
        self,
        methods: Sequence[GomsMethod],
        *,
        parallel: bool = True,
    ) -> None:
        """Build the operator DAG from selected methods.

        Parameters
        ----------
        methods : Sequence[GomsMethod]
            Methods to include.
        parallel : bool
            If True, methods on independent targets can execute in
            parallel (CPM-GOMS). If False, all methods are sequential.
        """
        self._nodes.clear()
        method_ranges: List[Tuple[int, int]] = []
        offset = 0

        for method in methods:
            start_idx = offset
            for i, op in enumerate(method.operators):
                node = OperatorNode(
                    index=offset + i,
                    operator=op,
                    method_id=method.method_id,
                    goal_id=method.goal_id,
                )
                if i > 0:
                    # Sequential within a method
                    prev_idx = offset + i - 1
                    node.predecessors.append(prev_idx)
                    self._nodes[prev_idx].successors.append(offset + i)
                self._nodes.append(node)
            end_idx = offset + len(method.operators)
            method_ranges.append((start_idx, end_idx))
            offset = end_idx

        if not parallel:
            # Chain all methods sequentially
            for i in range(1, len(method_ranges)):
                prev_end = method_ranges[i - 1][1] - 1
                curr_start = method_ranges[i][0]
                if prev_end >= 0 and curr_start < len(self._nodes):
                    self._nodes[curr_start].predecessors.append(prev_end)
                    self._nodes[prev_end].successors.append(curr_start)
        else:
            # Add dependencies for shared target resources
            target_to_methods: Dict[str, List[int]] = {}
            for mi, method in enumerate(methods):
                for op in method.operators:
                    if op.target_id:
                        target_to_methods.setdefault(op.target_id, []).append(mi)

            for _target, mi_list in target_to_methods.items():
                unique = sorted(set(mi_list))
                for k in range(len(unique) - 1):
                    prev_end = method_ranges[unique[k]][1] - 1
                    curr_start = method_ranges[unique[k + 1]][0]
                    if (
                        prev_end >= 0
                        and curr_start < len(self._nodes)
                        and curr_start not in self._nodes[prev_end].successors
                    ):
                        self._nodes[prev_end].successors.append(curr_start)
                        self._nodes[curr_start].predecessors.append(prev_end)

        self._compute_schedule()
        self._built = True

    # ── Forward / backward pass ───────────────────────────────────────

    def _compute_schedule(self) -> None:
        """Compute earliest/latest start/finish via CPM forward and backward pass."""
        n = len(self._nodes)
        if n == 0:
            self._makespan = 0.0
            return

        # Topological order (Kahn's algorithm)
        in_degree = [0] * n
        for node in self._nodes:
            for succ in node.successors:
                in_degree[succ] += 1

        queue: deque[int] = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)

        topo_order: List[int] = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in self._nodes[u].successors:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # Forward pass: earliest start / finish
        for idx in topo_order:
            node = self._nodes[idx]
            es = 0.0
            for pred in node.predecessors:
                es = max(es, self._nodes[pred].earliest_finish)
            node.earliest_start = es
            node.earliest_finish = es + node.duration_s

        self._makespan = max(
            (node.earliest_finish for node in self._nodes),
            default=0.0,
        )

        # Backward pass: latest start / finish
        for node in self._nodes:
            node.latest_finish = self._makespan
            node.latest_start = self._makespan

        for idx in reversed(topo_order):
            node = self._nodes[idx]
            if not node.successors:
                node.latest_finish = self._makespan
            else:
                node.latest_finish = min(
                    self._nodes[s].latest_start for s in node.successors
                )
            node.latest_start = node.latest_finish - node.duration_s

    # ── Critical path extraction ──────────────────────────────────────

    def get_critical_path(self) -> List[OperatorNode]:
        """Return the operators on the critical path, in order."""
        return [node for node in self._nodes if node.is_critical]

    def get_critical_path_time(self) -> float:
        """Return the critical path duration."""
        return self._makespan

    # ── Slack times ───────────────────────────────────────────────────

    def get_slack_times(self) -> List[Tuple[int, float]]:
        """Return (operator_index, slack_s) for all operators."""
        return [(node.index, node.slack) for node in self._nodes]

    def get_non_critical_operators(
        self,
        *,
        min_slack_s: float = 0.0,
    ) -> List[OperatorNode]:
        """Return operators with slack > min_slack_s."""
        return [
            node for node in self._nodes
            if node.slack > min_slack_s + 1e-9
        ]

    # ── Bottleneck identification ─────────────────────────────────────

    def identify_bottlenecks(
        self,
        *,
        threshold_fraction: float = 0.15,
    ) -> List[Bottleneck]:
        """Identify bottleneck operators in the network.

        Categories:
        - **critical_path**: On the critical path and consumes
          > threshold of total time.
        - **resource_contention**: Multiple methods contend for
          the same target.
        - **memory_overload**: High WM load inferred from M-operator
          density.
        - **motor_difficulty**: High Fitts' ID pointing operators.
        """
        bottlenecks: List[Bottleneck] = []
        if self._makespan <= 0:
            return bottlenecks

        # Critical path bottlenecks
        for node in self._nodes:
            if not node.is_critical:
                continue
            frac = node.duration_s / self._makespan
            if frac > threshold_fraction:
                bottlenecks.append(Bottleneck(
                    operator_index=node.index,
                    operator=node.operator,
                    bottleneck_type="critical_path",
                    severity=min(1.0, frac / threshold_fraction * 0.5),
                    description=(
                        f"Critical-path operator {node.operator.op_type.name}: "
                        f"{node.operator.description} "
                        f"({node.duration_s:.2f}s, {frac:.0%} of makespan)"
                    ),
                ))

        # Resource contention
        target_nodes: Dict[str, List[int]] = {}
        for node in self._nodes:
            if node.operator.target_id:
                target_nodes.setdefault(node.operator.target_id, []).append(node.index)
        for target_id, node_indices in target_nodes.items():
            if len(node_indices) > 2:
                total_contention_time = sum(
                    self._nodes[idx].duration_s for idx in node_indices
                )
                bottlenecks.append(Bottleneck(
                    operator_index=node_indices[0],
                    operator=self._nodes[node_indices[0]].operator,
                    bottleneck_type="resource_contention",
                    severity=min(1.0, len(node_indices) / 5.0),
                    description=(
                        f"Target {target_id} accessed by {len(node_indices)} "
                        f"operators (total {total_contention_time:.2f}s)"
                    ),
                ))

        # Motor difficulty (high Fitts' ID)
        for node in self._nodes:
            if node.operator.op_type == OperatorType.P:
                fitts_id = node.operator.parameters.get("fitts_id", 0)
                if fitts_id > 5.0:
                    bottlenecks.append(Bottleneck(
                        operator_index=node.index,
                        operator=node.operator,
                        bottleneck_type="motor_difficulty",
                        severity=min(1.0, fitts_id / 8.0),
                        description=(
                            f"High Fitts' ID ({fitts_id:.1f}) for pointing to "
                            f"{node.operator.target_id}"
                        ),
                    ))

        # Memory overload (M-operator density)
        window_size = 5
        for i in range(len(self._nodes)):
            window_end = min(i + window_size, len(self._nodes))
            m_count = sum(
                1 for j in range(i, window_end)
                if self._nodes[j].operator.op_type == OperatorType.M
            )
            if m_count >= 3:
                bottlenecks.append(Bottleneck(
                    operator_index=i,
                    operator=self._nodes[i].operator,
                    bottleneck_type="memory_overload",
                    severity=min(1.0, m_count / WORKING_MEMORY_CAPACITY.midpoint),
                    description=(
                        f"High M-operator density ({m_count} in window of "
                        f"{window_size}) starting at operator {i}"
                    ),
                ))
                break  # Only report first occurrence

        return bottlenecks

    # ── What-if analysis ──────────────────────────────────────────────

    def what_if(
        self,
        operator_index: int,
        new_duration_s: float,
    ) -> Dict[str, Any]:
        """Analyse the impact of changing one operator's duration.

        Returns the change in makespan and whether the critical path
        changes.

        Parameters
        ----------
        operator_index : int
            Index of the operator to modify.
        new_duration_s : float
            New duration in seconds.

        Returns
        -------
        dict
            Impact analysis results.
        """
        if operator_index < 0 or operator_index >= len(self._nodes):
            raise IndexError(f"Operator index {operator_index} out of range")

        old_duration = self._nodes[operator_index].duration_s
        old_makespan = self._makespan
        was_critical = self._nodes[operator_index].is_critical

        # Temporarily change the operator
        old_op = self._nodes[operator_index].operator
        self._nodes[operator_index].operator = GomsOperator(
            op_type=old_op.op_type,
            duration_s=new_duration_s,
            target_id=old_op.target_id,
            target_bounds=old_op.target_bounds,
            description=old_op.description,
            parameters=old_op.parameters,
        )

        # Recompute
        self._compute_schedule()
        new_makespan = self._makespan
        is_critical_now = self._nodes[operator_index].is_critical

        # Restore
        self._nodes[operator_index].operator = old_op
        self._compute_schedule()

        return {
            "operator_index": operator_index,
            "old_duration_s": old_duration,
            "new_duration_s": new_duration_s,
            "old_makespan_s": old_makespan,
            "new_makespan_s": new_makespan,
            "makespan_delta_s": new_makespan - old_makespan,
            "was_critical": was_critical,
            "is_critical_after": is_critical_now,
            "critical_path_changed": was_critical != is_critical_now,
        }

    def batch_what_if(
        self,
        changes: Sequence[Tuple[int, float]],
    ) -> List[Dict[str, Any]]:
        """Run what-if analysis for multiple operator changes.

        Each change is evaluated independently.
        """
        return [self.what_if(idx, dur) for idx, dur in changes]

    # ── Gantt schedule ────────────────────────────────────────────────

    def get_schedule(self) -> List[ScheduleEntry]:
        """Generate a Gantt-chart-like schedule for all operators."""
        entries: List[ScheduleEntry] = []
        for node in self._nodes:
            entries.append(ScheduleEntry(
                operator_index=node.index,
                operator=node.operator,
                method_id=node.method_id,
                channel=node.channel,
                start_s=node.earliest_start,
                end_s=node.earliest_finish,
                is_critical=node.is_critical,
                slack_s=node.slack,
            ))
        return entries

    def get_schedule_by_channel(self) -> Dict[str, List[ScheduleEntry]]:
        """Group schedule entries by processing channel."""
        by_channel: Dict[str, List[ScheduleEntry]] = {}
        for entry in self.get_schedule():
            by_channel.setdefault(entry.channel, []).append(entry)
        # Sort each channel by start time
        for channel in by_channel:
            by_channel[channel].sort(key=lambda e: e.start_s)
        return by_channel

    def utilization_by_channel(self) -> Dict[str, float]:
        """Compute channel utilization as fraction of makespan.

        Returns a mapping from channel name to utilization (0-1).
        """
        if self._makespan <= 0:
            return {}
        by_channel = self.get_schedule_by_channel()
        result: Dict[str, float] = {}
        for channel, entries in by_channel.items():
            busy_time = sum(e.duration_s for e in entries)
            result[channel] = busy_time / self._makespan
        return result


__all__ = [
    "Bottleneck",
    "CriticalPathAnalyzer",
    "OperatorNode",
    "ScheduleEntry",
]
