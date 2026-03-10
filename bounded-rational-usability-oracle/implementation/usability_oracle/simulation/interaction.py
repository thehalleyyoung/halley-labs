"""
usability_oracle.simulation.interaction — Interaction event model.

Defines the data model for individual interaction events and
sequences of events that form a complete task interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InteractionEvent:
    """A single user interaction event."""
    step: int = 0
    action_id: str = ""
    action_name: str = ""
    timestamp: float = 0.0
    motor_time: float = 0.0
    decision_time: float = 0.0
    error: bool = False
    position: tuple[float, float] = (0.0, 0.0)
    wm_load: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        return self.motor_time + self.decision_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "timestamp": self.timestamp,
            "motor_time": self.motor_time,
            "decision_time": self.decision_time,
            "error": self.error,
            "position": list(self.position),
            "wm_load": self.wm_load,
        }


@dataclass
class InteractionSequence:
    """A sequence of interaction events forming a complete task."""
    events: list[InteractionEvent] = field(default_factory=list)
    task_name: str = ""
    agent_config: dict[str, Any] = field(default_factory=dict)
    goal_reached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.events)

    @property
    def total_time(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].timestamp

    @property
    def n_errors(self) -> int:
        return sum(1 for e in self.events if e.error)

    @property
    def error_rate(self) -> float:
        return self.n_errors / max(self.n_steps, 1)

    @property
    def mean_motor_time(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.motor_time for e in self.events) / len(self.events)

    @property
    def mean_decision_time(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.decision_time for e in self.events) / len(self.events)

    @property
    def max_wm_load(self) -> int:
        if not self.events:
            return 0
        return max(e.wm_load for e in self.events)

    def summary(self) -> str:
        return (
            f"Task: {self.task_name}\n"
            f"  Steps: {self.n_steps}\n"
            f"  Total time: {self.total_time:.3f}s\n"
            f"  Errors: {self.n_errors} ({self.error_rate:.1%})\n"
            f"  Goal reached: {self.goal_reached}\n"
            f"  Mean motor: {self.mean_motor_time:.3f}s\n"
            f"  Mean decision: {self.mean_decision_time:.3f}s\n"
            f"  Max WM load: {self.max_wm_load}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "n_steps": self.n_steps,
            "total_time": self.total_time,
            "n_errors": self.n_errors,
            "goal_reached": self.goal_reached,
            "events": [e.to_dict() for e in self.events],
        }

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def time_per_step(self) -> list[float]:
        """Get the total time for each step."""
        return [e.total_time for e in self.events]

    def cumulative_time(self) -> list[float]:
        """Get cumulative time at each step."""
        return [e.timestamp for e in self.events]

    def action_sequence(self) -> list[str]:
        """Get the sequence of action names."""
        return [e.action_name for e in self.events]

    def error_positions(self) -> list[int]:
        """Get step indices where errors occurred."""
        return [e.step for e in self.events if e.error]

    def wm_load_curve(self) -> list[int]:
        """Get working memory load at each step."""
        return [e.wm_load for e in self.events]

    def filter_by_action(self, action_name: str) -> list[InteractionEvent]:
        """Get events for a specific action."""
        return [e for e in self.events if e.action_name == action_name]

    def segment_by_phase(self, phase_markers: list[str]) -> list["InteractionSequence"]:
        """Split sequence into phases based on action names."""
        segments: list[InteractionSequence] = []
        current_events: list[InteractionEvent] = []
        phase_idx = 0

        for event in self.events:
            current_events.append(event)
            if phase_idx < len(phase_markers) and event.action_name == phase_markers[phase_idx]:
                segments.append(InteractionSequence(
                    events=current_events,
                    task_name=f"Phase {phase_idx + 1}",
                ))
                current_events = []
                phase_idx += 1

        if current_events:
            segments.append(InteractionSequence(
                events=current_events,
                task_name=f"Phase {phase_idx + 1}",
            ))

        return segments
