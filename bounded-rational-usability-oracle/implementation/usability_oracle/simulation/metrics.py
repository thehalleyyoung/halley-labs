"""
usability_oracle.simulation.metrics — Simulation metrics collection.

Computes aggregate metrics from simulation runs including task completion
time distributions, error rates, efficiency scores, and cognitive load
profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.simulation.interaction import InteractionSequence
from usability_oracle.simulation.recorder import Recording


@dataclass
class SimulationMetricsSummary:
    """Summary of metrics across multiple simulation runs."""
    n_runs: int = 0
    success_rate: float = 0.0
    mean_time: float = 0.0
    std_time: float = 0.0
    median_time: float = 0.0
    p95_time: float = 0.0
    mean_steps: float = 0.0
    mean_errors: float = 0.0
    error_rate: float = 0.0
    mean_wm_load: float = 0.0
    max_wm_load: float = 0.0
    efficiency: float = 0.0
    throughput: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Simulation Metrics ({self.n_runs} runs):\n"
            f"  Success rate:   {self.success_rate:.1%}\n"
            f"  Time:           {self.mean_time:.3f}s ± {self.std_time:.3f}s\n"
            f"  Median time:    {self.median_time:.3f}s\n"
            f"  P95 time:       {self.p95_time:.3f}s\n"
            f"  Steps:          {self.mean_steps:.1f}\n"
            f"  Errors:         {self.mean_errors:.2f} ({self.error_rate:.1%})\n"
            f"  WM load:        {self.mean_wm_load:.1f} (max {self.max_wm_load:.0f})\n"
            f"  Efficiency:     {self.efficiency:.3f}\n"
            f"  Throughput:     {self.throughput:.3f} steps/s"
        )


class SimulationMetrics:
    """Compute metrics from simulation recordings."""

    # ------------------------------------------------------------------
    # From sequences
    # ------------------------------------------------------------------

    @staticmethod
    def from_sequences(sequences: list[InteractionSequence]) -> SimulationMetricsSummary:
        """Compute metrics from a list of interaction sequences."""
        if not sequences:
            return SimulationMetricsSummary()

        n = len(sequences)
        times = np.array([s.total_time for s in sequences])
        steps = np.array([s.n_steps for s in sequences])
        errors = np.array([s.n_errors for s in sequences])
        success = np.array([1.0 if s.goal_reached else 0.0 for s in sequences])

        wm_loads = []
        max_wm = 0
        for s in sequences:
            for e in s.events:
                wm_loads.append(e.wm_load)
                max_wm = max(max_wm, e.wm_load)

        wm_arr = np.array(wm_loads) if wm_loads else np.array([0])

        # Efficiency: ratio of optimal steps to actual steps
        min_steps = min(s.n_steps for s in sequences) if sequences else 1
        efficiency = min_steps / max(np.mean(steps), 1)

        # Throughput: steps per second
        total_time = float(np.sum(times))
        total_steps = int(np.sum(steps))
        throughput = total_steps / max(total_time, 0.001)

        return SimulationMetricsSummary(
            n_runs=n,
            success_rate=float(np.mean(success)),
            mean_time=float(np.mean(times)),
            std_time=float(np.std(times)),
            median_time=float(np.median(times)),
            p95_time=float(np.percentile(times, 95)),
            mean_steps=float(np.mean(steps)),
            mean_errors=float(np.mean(errors)),
            error_rate=float(np.sum(errors) / max(np.sum(steps), 1)),
            mean_wm_load=float(np.mean(wm_arr)),
            max_wm_load=float(max_wm),
            efficiency=efficiency,
            throughput=throughput,
        )

    # ------------------------------------------------------------------
    # From recording
    # ------------------------------------------------------------------

    @staticmethod
    def from_recording(recording: Recording) -> SimulationMetricsSummary:
        """Compute metrics from a recording."""
        return SimulationMetrics.from_sequences(recording.sequences)

    # ------------------------------------------------------------------
    # Comparative metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        baseline: list[InteractionSequence],
        treatment: list[InteractionSequence],
    ) -> dict[str, Any]:
        """Compare metrics between baseline and treatment conditions."""
        m_base = SimulationMetrics.from_sequences(baseline)
        m_treat = SimulationMetrics.from_sequences(treatment)

        def _pct_change(a: float, b: float) -> float:
            return (b - a) / max(abs(a), 0.001) * 100

        return {
            "baseline": m_base,
            "treatment": m_treat,
            "time_diff": m_treat.mean_time - m_base.mean_time,
            "time_pct_change": _pct_change(m_base.mean_time, m_treat.mean_time),
            "success_diff": m_treat.success_rate - m_base.success_rate,
            "error_rate_diff": m_treat.error_rate - m_base.error_rate,
            "efficiency_diff": m_treat.efficiency - m_base.efficiency,
            "improved": m_treat.mean_time < m_base.mean_time and m_treat.success_rate >= m_base.success_rate,
        }

    # ------------------------------------------------------------------
    # Time-series analysis
    # ------------------------------------------------------------------

    @staticmethod
    def learning_curve(sequences: list[InteractionSequence]) -> list[tuple[int, float]]:
        """Compute a learning curve (run_index, time) showing improvement over runs."""
        return [(i, s.total_time) for i, s in enumerate(sequences)]

    @staticmethod
    def step_time_distribution(sequences: list[InteractionSequence]) -> dict[str, np.ndarray]:
        """Get time distributions per step position."""
        max_steps = max(s.n_steps for s in sequences) if sequences else 0
        step_times: dict[int, list[float]] = {}

        for s in sequences:
            for e in s.events:
                step_times.setdefault(e.step, []).append(e.total_time)

        return {f"step_{k}": np.array(v) for k, v in sorted(step_times.items())}

    # ------------------------------------------------------------------
    # Cognitive load profile
    # ------------------------------------------------------------------

    @staticmethod
    def cognitive_load_profile(sequence: InteractionSequence) -> dict[str, Any]:
        """Analyse the cognitive load profile of a single sequence."""
        if not sequence.events:
            return {}

        motor_times = [e.motor_time for e in sequence.events]
        decision_times = [e.decision_time for e in sequence.events]
        wm_loads = [e.wm_load for e in sequence.events]

        motor_arr = np.array(motor_times)
        decision_arr = np.array(decision_times)
        wm_arr = np.array(wm_loads)

        # Find peak load moments
        peak_idx = int(np.argmax(wm_arr))
        peak_decision_idx = int(np.argmax(decision_arr))

        return {
            "mean_motor": float(motor_arr.mean()),
            "mean_decision": float(decision_arr.mean()),
            "motor_fraction": float(motor_arr.sum() / max(motor_arr.sum() + decision_arr.sum(), 0.001)),
            "decision_fraction": float(decision_arr.sum() / max(motor_arr.sum() + decision_arr.sum(), 0.001)),
            "peak_wm_step": peak_idx,
            "peak_wm_load": int(wm_arr[peak_idx]),
            "peak_decision_step": peak_decision_idx,
            "peak_decision_time": float(decision_arr[peak_decision_idx]),
            "wm_overload_steps": int(np.sum(wm_arr >= 4)),
        }
