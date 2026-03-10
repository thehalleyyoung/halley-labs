"""
usability_oracle.goms.simulation — GOMS execution simulation.

Discrete-event simulation of GOMS model execution with stochastic
operator times, error injection, working memory load tracking,
parallel operator execution for expert models, and execution trace
comparison for regression detection.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.core.constants import WORKING_MEMORY_CAPACITY
from usability_oracle.goms.types import (
    GomsGoal,
    GomsMethod,
    GomsModel,
    GomsOperator,
    GomsTrace,
    OperatorType,
)
from usability_oracle.goms.klm import KLMConfig, SkillLevel


# ═══════════════════════════════════════════════════════════════════════════
# Simulation event and trace types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SimEvent:
    """A single event in the discrete-event simulation."""

    timestamp_s: float
    end_s: float
    operator: GomsOperator
    method_id: str
    goal_id: str
    is_error: bool = False
    wm_load: float = 0.0
    """Working memory load (chunks) at event start."""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.timestamp_s


@dataclass
class SimTrace:
    """Complete simulation trace with events and summary statistics."""

    trace_id: str
    task_name: str
    events: List[SimEvent] = field(default_factory=list)
    total_time_s: float = 0.0
    error_count: int = 0
    peak_wm_load: float = 0.0
    parallel_savings_s: float = 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_goms_trace(self, goals: Sequence[GomsGoal], methods: Sequence[GomsMethod]) -> GomsTrace:
        """Convert simulation trace to GomsTrace for analysis."""
        return GomsTrace(
            trace_id=self.trace_id,
            task_name=self.task_name,
            goals=tuple(goals),
            methods_selected=tuple(methods),
            total_time_s=self.total_time_s,
            critical_path_time_s=self.total_time_s - self.parallel_savings_s,
            metadata={
                "error_count": self.error_count,
                "peak_wm_load": self.peak_wm_load,
                "parallel_savings_s": self.parallel_savings_s,
                "event_count": self.event_count,
                "simulated": True,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Stochastic duration models
# ═══════════════════════════════════════════════════════════════════════════

# Operator duration distributions (log-normal parameters):
# Each maps OperatorType → (mu_log, sigma_log) of the underlying normal.
# The mean of log-normal(μ, σ) = exp(μ + σ²/2).

def _lognormal_params(mean: float, cv: float) -> Tuple[float, float]:
    """Compute log-normal μ, σ from desired mean and coefficient of variation."""
    if mean <= 0:
        return (0.0, 0.0)
    sigma_sq = math.log(1.0 + cv * cv)
    mu = math.log(mean) - sigma_sq / 2.0
    return (mu, math.sqrt(sigma_sq))


# Coefficient of variation by operator type (from empirical data)
_CV_BY_TYPE: Dict[OperatorType, float] = {
    OperatorType.K: 0.20,
    OperatorType.P: 0.25,
    OperatorType.H: 0.15,
    OperatorType.D: 0.30,
    OperatorType.M: 0.35,
    OperatorType.R: 0.50,
    OperatorType.W: 0.50,
    OperatorType.B: 0.10,
}


def sample_duration(
    op: GomsOperator,
    rng: np.random.Generator,
) -> float:
    """Sample a stochastic duration for an operator.

    Uses a log-normal distribution centered on the operator's nominal
    duration with type-appropriate coefficient of variation.
    """
    mean = op.duration_s
    if mean <= 0:
        return 0.0
    cv = _CV_BY_TYPE.get(op.op_type, 0.20)
    mu, sigma = _lognormal_params(mean, cv)
    return float(rng.lognormal(mu, sigma))


# ═══════════════════════════════════════════════════════════════════════════
# Error injection
# ═══════════════════════════════════════════════════════════════════════════

def compute_error_probability(
    op: GomsOperator,
    wm_load: float,
    skill_level: SkillLevel,
) -> float:
    """Compute error probability for a single operator.

    Factors:
    - Base rate depends on operator type and skill level.
    - Working memory load increases cognitive error rates.
    - Pointing errors increase with Fitts' index of difficulty.
    """
    base_rates: Dict[OperatorType, float] = {
        OperatorType.K: 0.01,
        OperatorType.P: 0.02,
        OperatorType.H: 0.005,
        OperatorType.D: 0.03,
        OperatorType.M: 0.01,
        OperatorType.R: 0.0,
        OperatorType.W: 0.0,
        OperatorType.B: 0.005,
    }
    p = base_rates.get(op.op_type, 0.01)

    # Skill adjustment
    if skill_level == SkillLevel.NOVICE:
        p *= 2.0
    elif skill_level == SkillLevel.EXPERT:
        p *= 0.5

    # WM load effect on cognitive operators
    wm_capacity = WORKING_MEMORY_CAPACITY.midpoint
    if op.is_cognitive and wm_load > wm_capacity * 0.5:
        overload_factor = 1.0 + (wm_load - wm_capacity * 0.5) / wm_capacity
        p *= overload_factor

    # Fitts' difficulty effect on pointing
    if op.op_type == OperatorType.P:
        fitts_id = op.parameters.get("fitts_id", 3.0)
        p *= 1.0 + 0.05 * max(0, fitts_id - 3.0)

    return min(p, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Working memory load tracking
# ═══════════════════════════════════════════════════════════════════════════

class WorkingMemoryTracker:
    """Track working memory load during simulation.

    Each M operator adds a chunk; system responses allow decay;
    chunks decay exponentially over time.
    """

    def __init__(self, decay_half_life_s: float = 10.0) -> None:
        self._chunks: List[float] = []
        """Timestamps when each chunk was added."""
        self._half_life = decay_half_life_s
        self._current_time = 0.0

    @property
    def current_load(self) -> float:
        """Current effective WM load in chunks."""
        if not self._chunks:
            return 0.0
        load = 0.0
        for t_added in self._chunks:
            elapsed = self._current_time - t_added
            # Exponential decay: P(recall) = exp(-λt), λ = ln2 / half_life
            if self._half_life > 0:
                decay = math.exp(-math.log(2) * elapsed / self._half_life)
            else:
                decay = 1.0
            load += decay
        return load

    def add_chunk(self, timestamp_s: float) -> None:
        """Record a new WM chunk at the given timestamp."""
        self._current_time = max(self._current_time, timestamp_s)
        self._chunks.append(timestamp_s)

    def advance_time(self, timestamp_s: float) -> None:
        """Advance the internal clock (for decay computation)."""
        self._current_time = max(self._current_time, timestamp_s)

    def on_system_response(self, duration_s: float) -> None:
        """Model WM decay during system response wait."""
        self._current_time += duration_s

    def reset(self) -> None:
        self._chunks.clear()
        self._current_time = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Parallel operator scheduling (for expert CPM-GOMS)
# ═══════════════════════════════════════════════════════════════════════════

def schedule_parallel_execution(
    methods: Sequence[GomsMethod],
    *,
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE,
) -> Tuple[List[Tuple[int, int, float, float]], float]:
    """Schedule operators with parallel execution for expert models.

    In CPM-GOMS, cognitive, perceptual, and motor processors can
    operate in parallel. For novice/intermediate, everything is serial.

    Returns
    -------
    tuple[list[tuple[int, int, float, float]], float]
        Schedule as (method_idx, op_idx, start_s, end_s) and total time.
    """
    schedule: List[Tuple[int, int, float, float]] = []

    if skill_level != SkillLevel.EXPERT:
        # Serial execution
        clock = 0.0
        for mi, method in enumerate(methods):
            for oi, op in enumerate(method.operators):
                start = clock
                end = clock + op.duration_s
                schedule.append((mi, oi, start, end))
                clock = end
        return schedule, clock

    # Expert parallel execution: track channel availability
    cognitive_free = 0.0  # When cognitive processor is free
    motor_free = 0.0      # When motor processor is free
    perceptual_free = 0.0  # When perceptual processor is free
    system_free = 0.0

    for mi, method in enumerate(methods):
        for oi, op in enumerate(method.operators):
            if op.is_cognitive:
                start = cognitive_free
                end = start + op.duration_s
                cognitive_free = end
            elif op.is_motor:
                # Motor depends on preceding cognitive being done
                start = max(motor_free, cognitive_free)
                end = start + op.duration_s
                motor_free = end
            elif op.op_type.is_system:
                start = max(motor_free, cognitive_free)
                end = start + op.duration_s
                system_free = end
                # After system response, user needs to re-perceive
                perceptual_free = end
            else:
                # Default: wait for all channels
                start = max(cognitive_free, motor_free, perceptual_free, system_free)
                end = start + op.duration_s

            schedule.append((mi, oi, start, end))

    total_time = max(
        (end for _, _, _, end in schedule),
        default=0.0,
    )
    return schedule, total_time


# ═══════════════════════════════════════════════════════════════════════════
# Main simulation engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """Configuration for GOMS simulation."""

    klm_config: KLMConfig = field(default_factory=KLMConfig)
    stochastic: bool = True
    """Whether to use stochastic operator times."""
    inject_errors: bool = True
    """Whether to inject errors based on complexity."""
    error_recovery_time_s: float = 2.0
    """Time cost for recovering from an error."""
    seed: int = 42
    wm_decay_half_life_s: float = 10.0


class GomsSimulator:
    """Discrete-event simulator for GOMS model execution."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self._config = config or SimulationConfig()
        self._rng = np.random.default_rng(self._config.seed)

    def simulate(
        self,
        model: GomsModel,
        methods: Sequence[GomsMethod],
        *,
        n_runs: int = 1,
    ) -> List[SimTrace]:
        """Simulate GOMS model execution.

        Parameters
        ----------
        model : GomsModel
            The GOMS model.
        methods : Sequence[GomsMethod]
            Selected methods to execute.
        n_runs : int
            Number of simulation runs (for Monte Carlo analysis).

        Returns
        -------
        list[SimTrace]
            One SimTrace per run.
        """
        traces: List[SimTrace] = []
        for run_idx in range(n_runs):
            trace = self._run_once(model, methods, run_idx)
            traces.append(trace)
        return traces

    def _run_once(
        self,
        model: GomsModel,
        methods: Sequence[GomsMethod],
        run_idx: int,
    ) -> SimTrace:
        """Execute a single simulation run."""
        skill = self._config.klm_config.skill_level
        trace = SimTrace(
            trace_id=f"sim-{uuid.uuid4().hex[:8]}-r{run_idx}",
            task_name=model.name,
        )
        wm = WorkingMemoryTracker(
            decay_half_life_s=self._config.wm_decay_half_life_s,
        )

        # Schedule operators (parallel for experts)
        schedule, _ = schedule_parallel_execution(
            methods, skill_level=skill,
        )

        # Build operator lookup
        all_ops: List[Tuple[GomsMethod, GomsOperator]] = []
        for method in methods:
            for op in method.operators:
                all_ops.append((method, op))

        clock = 0.0
        peak_wm = 0.0
        error_count = 0
        serial_time = sum(op.duration_s for m in methods for op in m.operators)

        for sched_idx, (mi, oi, sched_start, sched_end) in enumerate(schedule):
            method = methods[mi]
            op = method.operators[oi]

            # Determine actual duration
            if self._config.stochastic:
                actual_duration = sample_duration(op, self._rng)
            else:
                actual_duration = op.duration_s

            start_time = sched_start
            end_time = start_time + actual_duration

            # Update working memory
            wm.advance_time(start_time)
            current_wm = wm.current_load
            peak_wm = max(peak_wm, current_wm)

            if op.is_cognitive:
                wm.add_chunk(start_time)
            elif op.op_type == OperatorType.R:
                wm.on_system_response(actual_duration)

            # Error injection
            is_error = False
            if self._config.inject_errors:
                p_err = compute_error_probability(op, current_wm, skill)
                if self._rng.random() < p_err:
                    is_error = True
                    error_count += 1
                    # Error recovery adds time
                    end_time += self._config.error_recovery_time_s

            trace.events.append(SimEvent(
                timestamp_s=start_time,
                end_s=end_time,
                operator=op,
                method_id=method.method_id,
                goal_id=method.goal_id,
                is_error=is_error,
                wm_load=current_wm,
            ))

        if trace.events:
            trace.total_time_s = max(e.end_s for e in trace.events)
        trace.error_count = error_count
        trace.peak_wm_load = peak_wm
        trace.parallel_savings_s = max(0, serial_time - trace.total_time_s)

        return trace

    def compare_versions(
        self,
        model_old: GomsModel,
        methods_old: Sequence[GomsMethod],
        model_new: GomsModel,
        methods_new: Sequence[GomsMethod],
        *,
        n_runs: int = 30,
    ) -> Dict[str, Any]:
        """Compare simulation results across UI versions.

        Runs Monte Carlo simulations on both versions and reports
        statistical comparison for regression detection.
        """
        traces_old = self.simulate(model_old, methods_old, n_runs=n_runs)
        traces_new = self.simulate(model_new, methods_new, n_runs=n_runs)

        times_old = np.array([t.total_time_s for t in traces_old])
        times_new = np.array([t.total_time_s for t in traces_new])
        errors_old = np.array([t.error_count for t in traces_old], dtype=float)
        errors_new = np.array([t.error_count for t in traces_new], dtype=float)

        # Welch's t-test approximation
        mean_diff = float(times_new.mean() - times_old.mean())
        pooled_se = math.sqrt(
            times_old.var(ddof=1) / len(times_old)
            + times_new.var(ddof=1) / len(times_new)
        ) if len(times_old) > 1 else 1.0

        t_stat = mean_diff / pooled_se if pooled_se > 0 else 0.0

        # Cohen's d effect size
        pooled_std = math.sqrt(
            (times_old.var(ddof=1) + times_new.var(ddof=1)) / 2
        ) if len(times_old) > 1 else 1.0
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        return {
            "mean_time_old_s": float(times_old.mean()),
            "mean_time_new_s": float(times_new.mean()),
            "time_delta_s": mean_diff,
            "t_statistic": t_stat,
            "cohens_d": cohens_d,
            "mean_errors_old": float(errors_old.mean()),
            "mean_errors_new": float(errors_new.mean()),
            "regression_detected": mean_diff > 0.5 and t_stat > 2.0,
            "n_runs": n_runs,
        }


__all__ = [
    "GomsSimulator",
    "SimEvent",
    "SimTrace",
    "SimulationConfig",
    "WorkingMemoryTracker",
    "compute_error_probability",
    "sample_duration",
    "schedule_parallel_execution",
]
