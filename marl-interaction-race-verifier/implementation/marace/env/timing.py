"""
Configurable timing and latency models for multi-agent environments.

Provides abstract and concrete timing models, hardware profiles,
latency scheduling, jitter simulation, synchronization barriers,
and timing analysis utilities.
"""

from __future__ import annotations

import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Abstract timing model
# ---------------------------------------------------------------------------

class TimingModel(ABC):
    """Abstract timing model for agent computation latency.

    A timing model maps ``(agent_id, tick)`` to a non-negative latency
    value (in seconds).  Concrete subclasses implement the sampling
    strategy.
    """

    @abstractmethod
    def sample(self, agent_id: str, tick: int) -> float:
        """Sample the latency for *agent_id* at *tick*.

        Returns:
            Non-negative latency in seconds.
        """

    @abstractmethod
    def mean(self, agent_id: str) -> float:
        """Expected latency for *agent_id*."""

    @abstractmethod
    def worst_case(self, agent_id: str) -> float:
        """Worst-case (upper-bound) latency for *agent_id*."""

    def reset(self) -> None:
        """Reset internal RNG state (if any)."""


# ---------------------------------------------------------------------------
# Fixed latency
# ---------------------------------------------------------------------------

class FixedLatencyModel(TimingModel):
    """Constant latency per agent.

    Attributes:
        latencies: Mapping from agent id to fixed latency (seconds).
        default_latency: Latency for agents not in *latencies*.
    """

    def __init__(
        self,
        latencies: Optional[Dict[str, float]] = None,
        default_latency: float = 0.0,
    ) -> None:
        self._latencies = dict(latencies or {})
        self._default = default_latency

    def sample(self, agent_id: str, tick: int) -> float:
        return self._latencies.get(agent_id, self._default)

    def mean(self, agent_id: str) -> float:
        return self._latencies.get(agent_id, self._default)

    def worst_case(self, agent_id: str) -> float:
        return self._latencies.get(agent_id, self._default)

    def set_latency(self, agent_id: str, latency: float) -> None:
        if latency < 0:
            raise ValueError("Latency must be non-negative")
        self._latencies[agent_id] = latency

    def __repr__(self) -> str:
        return f"FixedLatencyModel(latencies={self._latencies}, default={self._default})"


# ---------------------------------------------------------------------------
# Stochastic latency
# ---------------------------------------------------------------------------

class DistributionType(enum.Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    LOGNORMAL = "lognormal"


@dataclass
class LatencyDistribution:
    """Parameters for a stochastic latency distribution.

    Attributes:
        dist_type: Distribution family.
        mean_latency: Mean latency (seconds).
        std_latency: Standard deviation (used by NORMAL, LOGNORMAL).
        low: Lower bound (used by UNIFORM).
        high: Upper bound (used by UNIFORM).
    """
    dist_type: DistributionType = DistributionType.NORMAL
    mean_latency: float = 0.01
    std_latency: float = 0.002
    low: float = 0.005
    high: float = 0.02


class StochasticLatencyModel(TimingModel):
    """Latency drawn from a configurable probability distribution.

    Each agent may have its own distribution parameters.  Samples are
    clamped to ``[0, +∞)``.
    """

    def __init__(
        self,
        distributions: Optional[Dict[str, LatencyDistribution]] = None,
        default_dist: Optional[LatencyDistribution] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._distributions = dict(distributions or {})
        self._default = default_dist or LatencyDistribution()
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def _get_dist(self, agent_id: str) -> LatencyDistribution:
        return self._distributions.get(agent_id, self._default)

    def sample(self, agent_id: str, tick: int) -> float:
        d = self._get_dist(agent_id)
        if d.dist_type == DistributionType.NORMAL:
            val = self._rng.normal(d.mean_latency, d.std_latency)
        elif d.dist_type == DistributionType.UNIFORM:
            val = self._rng.uniform(d.low, d.high)
        elif d.dist_type == DistributionType.EXPONENTIAL:
            val = self._rng.exponential(d.mean_latency)
        elif d.dist_type == DistributionType.LOGNORMAL:
            sigma2 = math.log(1 + (d.std_latency / d.mean_latency) ** 2)
            mu = math.log(d.mean_latency) - sigma2 / 2
            val = self._rng.lognormal(mu, math.sqrt(sigma2))
        else:
            val = d.mean_latency
        return max(0.0, val)

    def mean(self, agent_id: str) -> float:
        return self._get_dist(agent_id).mean_latency

    def worst_case(self, agent_id: str) -> float:
        d = self._get_dist(agent_id)
        if d.dist_type == DistributionType.UNIFORM:
            return d.high
        # 3-sigma upper bound
        return d.mean_latency + 3.0 * d.std_latency

    def set_distribution(self, agent_id: str, dist: LatencyDistribution) -> None:
        self._distributions[agent_id] = dist

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def __repr__(self) -> str:
        return (
            f"StochasticLatencyModel(num_agents={len(self._distributions)}, "
            f"default={self._default})"
        )


# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

class HardwareClass(enum.Enum):
    """Predefined hardware performance classes."""
    EMBEDDED_LOW = "embedded_low"
    EMBEDDED_HIGH = "embedded_high"
    DESKTOP = "desktop"
    SERVER = "server"
    GPU_ACCELERATED = "gpu_accelerated"
    EDGE_TPU = "edge_tpu"


_HARDWARE_PROFILES: Dict[HardwareClass, Dict[str, float]] = {
    HardwareClass.EMBEDDED_LOW: {
        "perception_latency": 0.050,
        "compute_latency": 0.100,
        "actuation_latency": 0.020,
        "jitter_std": 0.010,
    },
    HardwareClass.EMBEDDED_HIGH: {
        "perception_latency": 0.020,
        "compute_latency": 0.040,
        "actuation_latency": 0.010,
        "jitter_std": 0.005,
    },
    HardwareClass.DESKTOP: {
        "perception_latency": 0.010,
        "compute_latency": 0.020,
        "actuation_latency": 0.005,
        "jitter_std": 0.002,
    },
    HardwareClass.SERVER: {
        "perception_latency": 0.005,
        "compute_latency": 0.010,
        "actuation_latency": 0.002,
        "jitter_std": 0.001,
    },
    HardwareClass.GPU_ACCELERATED: {
        "perception_latency": 0.002,
        "compute_latency": 0.005,
        "actuation_latency": 0.002,
        "jitter_std": 0.001,
    },
    HardwareClass.EDGE_TPU: {
        "perception_latency": 0.003,
        "compute_latency": 0.008,
        "actuation_latency": 0.003,
        "jitter_std": 0.001,
    },
}


class HardwareProfile:
    """Predefined timing profiles for different hardware classes.

    Usage::

        profile = HardwareProfile(HardwareClass.EMBEDDED_LOW)
        config = profile.to_agent_config("agent_0")
    """

    def __init__(self, hw_class: HardwareClass) -> None:
        self.hw_class = hw_class
        self._params = dict(_HARDWARE_PROFILES[hw_class])

    @property
    def perception_latency(self) -> float:
        return self._params["perception_latency"]

    @property
    def compute_latency(self) -> float:
        return self._params["compute_latency"]

    @property
    def actuation_latency(self) -> float:
        return self._params["actuation_latency"]

    @property
    def jitter_std(self) -> float:
        return self._params["jitter_std"]

    @property
    def total_latency(self) -> float:
        return (
            self.perception_latency + self.compute_latency + self.actuation_latency
        )

    def to_agent_config(self, agent_id: str) -> "AgentTimingConfig":
        from marace.env.base import AgentTimingConfig

        return AgentTimingConfig(
            agent_id=agent_id,
            perception_latency=self.perception_latency,
            compute_latency=self.compute_latency,
            actuation_latency=self.actuation_latency,
            jitter_std=self.jitter_std,
        )

    def to_latency_distribution(self) -> LatencyDistribution:
        return LatencyDistribution(
            dist_type=DistributionType.NORMAL,
            mean_latency=self.total_latency,
            std_latency=self.jitter_std,
        )

    @staticmethod
    def available_profiles() -> List[HardwareClass]:
        return list(_HARDWARE_PROFILES.keys())

    def __repr__(self) -> str:
        return f"HardwareProfile({self.hw_class.value})"


# ---------------------------------------------------------------------------
# Latency scheduler
# ---------------------------------------------------------------------------

@dataclass
class ScheduledAction:
    """An action scheduled for future execution."""
    agent_id: str
    action: Any
    submit_time: float
    execute_time: float
    latency: float


class LatencyScheduler:
    """Schedule agent actions based on their latencies.

    The scheduler maintains a priority queue of pending actions ordered by
    their *execute_time* (submit_time + latency).  Actions become eligible
    for execution when the environment clock reaches their execute_time.
    """

    def __init__(self, timing_model: TimingModel) -> None:
        self._model = timing_model
        self._queue: List[ScheduledAction] = []
        self._executed: List[ScheduledAction] = []
        self._tick: int = 0

    def submit(
        self, agent_id: str, action: Any, current_time: float
    ) -> ScheduledAction:
        """Submit an action for scheduled execution.

        Returns:
            The ``ScheduledAction`` describing when the action will execute.
        """
        self._tick += 1
        latency = self._model.sample(agent_id, self._tick)
        sa = ScheduledAction(
            agent_id=agent_id,
            action=action,
            submit_time=current_time,
            execute_time=current_time + latency,
            latency=latency,
        )
        self._queue.append(sa)
        self._queue.sort(key=lambda s: s.execute_time)
        return sa

    def pending(self) -> List[ScheduledAction]:
        """Return all pending actions in execution order."""
        return list(self._queue)

    def pop_ready(self, current_time: float) -> List[ScheduledAction]:
        """Pop all actions whose execute_time ≤ *current_time*."""
        ready: List[ScheduledAction] = []
        remaining: List[ScheduledAction] = []
        for sa in self._queue:
            if sa.execute_time <= current_time:
                ready.append(sa)
                self._executed.append(sa)
            else:
                remaining.append(sa)
        self._queue = remaining
        return ready

    def peek_next_time(self) -> Optional[float]:
        """Return the execute_time of the next pending action, or None."""
        if self._queue:
            return self._queue[0].execute_time
        return None

    @property
    def history(self) -> List[ScheduledAction]:
        return list(self._executed)

    def clear(self) -> None:
        self._queue.clear()
        self._executed.clear()
        self._tick = 0

    def __repr__(self) -> str:
        return f"LatencyScheduler(pending={len(self._queue)}, executed={len(self._executed)})"


# ---------------------------------------------------------------------------
# Timing jitter
# ---------------------------------------------------------------------------

class TimingJitter:
    """Add timing jitter to simulate real-world variation.

    Wraps an existing ``TimingModel`` and adds a secondary jitter term
    drawn from a specified distribution.
    """

    def __init__(
        self,
        base_model: TimingModel,
        jitter_dist: DistributionType = DistributionType.NORMAL,
        jitter_scale: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        self._base = base_model
        self._dist = jitter_dist
        self._scale = jitter_scale
        self._rng = np.random.default_rng(seed)

    def sample(self, agent_id: str, tick: int) -> float:
        base_val = self._base.sample(agent_id, tick)
        if self._dist == DistributionType.NORMAL:
            jitter = self._rng.normal(0.0, self._scale)
        elif self._dist == DistributionType.UNIFORM:
            jitter = self._rng.uniform(-self._scale, self._scale)
        elif self._dist == DistributionType.EXPONENTIAL:
            jitter = self._rng.exponential(self._scale)
        else:
            jitter = self._rng.normal(0.0, self._scale)
        return max(0.0, base_val + jitter)

    def mean(self, agent_id: str) -> float:
        return self._base.mean(agent_id)

    def worst_case(self, agent_id: str) -> float:
        return self._base.worst_case(agent_id) + 3.0 * self._scale

    def __repr__(self) -> str:
        return (
            f"TimingJitter(base={self._base!r}, "
            f"dist={self._dist.value}, scale={self._scale})"
        )


# ---------------------------------------------------------------------------
# Synchronization barrier
# ---------------------------------------------------------------------------

class SynchronizationBarrier:
    """Optional synchronization points in async execution.

    A barrier collects agent completions and releases them simultaneously
    once all expected agents have arrived.

    Attributes:
        agent_ids: Agents expected at this barrier.
        name: Human-readable label.
    """

    def __init__(self, agent_ids: List[str], name: str = "barrier") -> None:
        self.agent_ids = list(agent_ids)
        self.name = name
        self._arrived: Dict[str, float] = {}
        self._release_count: int = 0

    def arrive(self, agent_id: str, timestamp: float) -> bool:
        """Record *agent_id* arriving at the barrier.

        Returns:
            ``True`` if all agents have arrived and the barrier is released.
        """
        if agent_id not in self.agent_ids:
            raise ValueError(f"{agent_id} is not part of barrier {self.name}")
        self._arrived[agent_id] = timestamp
        if self.is_complete:
            self._release_count += 1
            return True
        return False

    @property
    def is_complete(self) -> bool:
        return all(a in self._arrived for a in self.agent_ids)

    @property
    def waiting(self) -> List[str]:
        return [a for a in self.agent_ids if a not in self._arrived]

    @property
    def max_wait_time(self) -> float:
        """Maximum wait time experienced by any agent at this barrier."""
        if not self._arrived:
            return 0.0
        times = list(self._arrived.values())
        return max(times) - min(times)

    def reset(self) -> None:
        self._arrived.clear()

    @property
    def release_count(self) -> int:
        return self._release_count

    def __repr__(self) -> str:
        return (
            f"SynchronizationBarrier({self.name!r}, "
            f"arrived={len(self._arrived)}/{len(self.agent_ids)})"
        )


# ---------------------------------------------------------------------------
# Timing analyser
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    """Summary statistics for an agent's timing behaviour."""
    agent_id: str
    num_samples: int
    mean_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float


class TimingAnalyzer:
    """Analyse timing properties of an execution trace.

    Collects per-agent latency samples from a ``LatencyScheduler`` or
    manual submissions and computes summary statistics.
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}

    def record(self, agent_id: str, latency: float) -> None:
        """Record a latency observation."""
        self._samples.setdefault(agent_id, []).append(latency)

    def ingest_scheduler(self, scheduler: LatencyScheduler) -> None:
        """Import timing data from a ``LatencyScheduler``."""
        for sa in scheduler.history:
            self.record(sa.agent_id, sa.latency)

    def stats(self, agent_id: str) -> TimingStats:
        """Compute summary statistics for *agent_id*."""
        data = np.array(self._samples.get(agent_id, []))
        if len(data) == 0:
            return TimingStats(
                agent_id=agent_id,
                num_samples=0,
                mean_latency=0.0,
                std_latency=0.0,
                min_latency=0.0,
                max_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
            )
        return TimingStats(
            agent_id=agent_id,
            num_samples=len(data),
            mean_latency=float(np.mean(data)),
            std_latency=float(np.std(data)),
            min_latency=float(np.min(data)),
            max_latency=float(np.max(data)),
            p50_latency=float(np.percentile(data, 50)),
            p95_latency=float(np.percentile(data, 95)),
            p99_latency=float(np.percentile(data, 99)),
        )

    def all_stats(self) -> Dict[str, TimingStats]:
        """Compute statistics for every recorded agent."""
        return {aid: self.stats(aid) for aid in self._samples}

    def detect_outliers(
        self, agent_id: str, z_threshold: float = 3.0
    ) -> List[Tuple[int, float]]:
        """Return (index, value) pairs for outlier latencies."""
        data = np.array(self._samples.get(agent_id, []))
        if len(data) < 2:
            return []
        mu = np.mean(data)
        sigma = np.std(data)
        if sigma == 0:
            return []
        z_scores = np.abs((data - mu) / sigma)
        outliers = [(int(i), float(data[i])) for i in np.where(z_scores > z_threshold)[0]]
        return outliers

    def ordering_violations(
        self,
        scheduler: LatencyScheduler,
    ) -> List[Tuple[ScheduledAction, ScheduledAction]]:
        """Find action pairs where submission order differs from execution order.

        These represent potential race conditions due to timing variations.
        """
        history = scheduler.history
        violations: List[Tuple[ScheduledAction, ScheduledAction]] = []
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                a, b = history[i], history[j]
                if a.submit_time < b.submit_time and a.execute_time > b.execute_time:
                    violations.append((a, b))
                elif a.submit_time > b.submit_time and a.execute_time < b.execute_time:
                    violations.append((b, a))
        return violations

    def max_reordering_window(self, scheduler: LatencyScheduler) -> float:
        """Maximum time window within which action reordering can occur."""
        violations = self.ordering_violations(scheduler)
        if not violations:
            return 0.0
        return max(
            abs(a.execute_time - b.execute_time) for a, b in violations
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export analysis results."""
        return {
            aid: {
                "stats": {
                    "mean": s.mean_latency,
                    "std": s.std_latency,
                    "min": s.min_latency,
                    "max": s.max_latency,
                    "p50": s.p50_latency,
                    "p95": s.p95_latency,
                    "p99": s.p99_latency,
                    "n": s.num_samples,
                },
                "outliers": self.detect_outliers(aid),
            }
            for aid, s in self.all_stats().items()
        }

    def clear(self) -> None:
        self._samples.clear()

    def __repr__(self) -> str:
        return f"TimingAnalyzer(agents={list(self._samples.keys())})"
