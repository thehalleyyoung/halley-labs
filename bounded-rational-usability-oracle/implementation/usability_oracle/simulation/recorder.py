"""
usability_oracle.simulation.recorder — Simulation recording and replay.

Records simulation runs for analysis, comparison, and replay.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.simulation.interaction import InteractionEvent, InteractionSequence


@dataclass
class Recording:
    """A complete recording of a simulation run."""
    id: str = ""
    timestamp: str = ""
    task_name: str = ""
    agent_config: dict[str, Any] = field(default_factory=dict)
    environment_config: dict[str, Any] = field(default_factory=dict)
    sequences: list[InteractionSequence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.sequences)

    @property
    def mean_time(self) -> float:
        if not self.sequences:
            return 0.0
        return sum(s.total_time for s in self.sequences) / len(self.sequences)

    @property
    def success_rate(self) -> float:
        if not self.sequences:
            return 0.0
        return sum(1 for s in self.sequences if s.goal_reached) / len(self.sequences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "task_name": self.task_name,
            "agent_config": self.agent_config,
            "n_runs": self.n_runs,
            "mean_time": self.mean_time,
            "success_rate": self.success_rate,
            "sequences": [s.to_dict() for s in self.sequences],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_json(data: str) -> "Recording":
        d = json.loads(data)
        sequences = []
        for sd in d.get("sequences", []):
            events = [InteractionEvent(**e) if isinstance(e, dict) else e for e in sd.get("events", [])]
            sequences.append(InteractionSequence(
                events=events,
                task_name=sd.get("task_name", ""),
                goal_reached=sd.get("goal_reached", False),
            ))
        return Recording(
            id=d.get("id", ""),
            timestamp=d.get("timestamp", ""),
            task_name=d.get("task_name", ""),
            agent_config=d.get("agent_config", {}),
            sequences=sequences,
            metadata=d.get("metadata", {}),
        )

    def summary(self) -> str:
        lines = [
            f"Recording: {self.task_name}",
            f"  Runs:         {self.n_runs}",
            f"  Success rate: {self.success_rate:.1%}",
            f"  Mean time:    {self.mean_time:.3f}s",
        ]
        if self.sequences:
            times = [s.total_time for s in self.sequences]
            import numpy as np
            arr = np.array(times)
            lines.extend([
                f"  Min time:     {arr.min():.3f}s",
                f"  Max time:     {arr.max():.3f}s",
                f"  Std time:     {arr.std():.3f}s",
            ])
        return "\n".join(lines)


class SimulationRecorder:
    """Record and manage simulation runs."""

    def __init__(self) -> None:
        self._recordings: dict[str, Recording] = {}
        self._active_recording: Optional[Recording] = None
        self._run_counter = 0

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def start_recording(
        self,
        task_name: str = "",
        agent_config: dict[str, Any] | None = None,
        environment_config: dict[str, Any] | None = None,
    ) -> str:
        """Start a new recording session. Returns recording ID."""
        self._run_counter += 1
        rec_id = f"rec-{self._run_counter:04d}"

        self._active_recording = Recording(
            id=rec_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            task_name=task_name,
            agent_config=agent_config or {},
            environment_config=environment_config or {},
        )
        return rec_id

    def record_sequence(self, sequence: InteractionSequence) -> None:
        """Add a completed interaction sequence to the active recording."""
        if self._active_recording is None:
            raise RuntimeError("No active recording. Call start_recording first.")
        self._active_recording.sequences.append(sequence)

    def stop_recording(self) -> Recording:
        """Stop and save the active recording."""
        if self._active_recording is None:
            raise RuntimeError("No active recording.")
        rec = self._active_recording
        self._recordings[rec.id] = rec
        self._active_recording = None
        return rec

    # ------------------------------------------------------------------
    # Access recordings
    # ------------------------------------------------------------------

    def get_recording(self, rec_id: str) -> Optional[Recording]:
        return self._recordings.get(rec_id)

    def list_recordings(self) -> list[str]:
        return list(self._recordings.keys())

    def compare_recordings(
        self,
        rec_id_a: str,
        rec_id_b: str,
    ) -> dict[str, Any]:
        """Compare two recordings."""
        a = self._recordings.get(rec_id_a)
        b = self._recordings.get(rec_id_b)
        if not a or not b:
            return {"error": "Recording not found"}

        return {
            "task_a": a.task_name,
            "task_b": b.task_name,
            "success_rate_a": a.success_rate,
            "success_rate_b": b.success_rate,
            "mean_time_a": a.mean_time,
            "mean_time_b": b.mean_time,
            "time_diff": b.mean_time - a.mean_time,
            "time_pct_change": (b.mean_time - a.mean_time) / max(a.mean_time, 0.001) * 100,
            "success_diff": b.success_rate - a.success_rate,
        }

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_all(self) -> str:
        """Export all recordings as JSON."""
        data = {rid: rec.to_dict() for rid, rec in self._recordings.items()}
        return json.dumps(data, indent=2)

    def import_recording(self, json_data: str) -> str:
        """Import a recording from JSON. Returns the recording ID."""
        rec = Recording.from_json(json_data)
        self._recordings[rec.id] = rec
        return rec.id
