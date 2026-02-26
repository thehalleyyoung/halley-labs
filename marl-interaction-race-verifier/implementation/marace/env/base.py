"""
Multi-agent environment interface and core abstractions.

Provides the abstract base class for multi-agent environments with support
for both synchronous and asynchronous stepping semantics, configurable
per-agent timing, and observation staleness modeling.
"""

from __future__ import annotations

import copy
import enum
import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvironmentState:
    """Base class for environment state with copy/hash support.

    Subclasses should store all mutable state as numpy arrays or plain
    Python objects so that ``copy()`` produces a deep, independent clone.
    """

    def copy(self) -> "EnvironmentState":
        """Return a deep copy of this state."""
        return copy.deepcopy(self)

    def fingerprint(self) -> str:
        """Return a deterministic hash of this state.

        The default implementation pickles the object and hashes the result.
        Subclasses may override for efficiency.
        """
        raw = pickle.dumps(self.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(raw).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvironmentState):
            return NotImplemented
        return self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return int(self.fingerprint()[:16], 16)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a JSON-compatible dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = copy.deepcopy(value)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentState":
        """Deserialize state from a dictionary."""
        obj = cls.__new__(cls)
        for key, value in data.items():
            if isinstance(value, list):
                setattr(obj, key, np.array(value))
            else:
                setattr(obj, key, value)
        return obj


# ---------------------------------------------------------------------------
# Stepping semantics
# ---------------------------------------------------------------------------

class SteppingOrder(enum.Enum):
    """Determines the order in which agents are stepped in async mode."""
    FIXED = "fixed"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    TIMING_BASED = "timing_based"


@dataclass
class AsyncSteppingSemantics:
    """Configuration for asynchronous stepping behaviour.

    Attributes:
        order: How to determine the order of agent steps.
        allow_simultaneous: Whether multiple agents may step at the same
            logical clock tick.
        max_steps_per_tick: Upper bound on the number of agent steps that
            may be executed within a single logical tick.
        fixed_order: Explicit ordering when ``order`` is ``FIXED``.
        rng_seed: Random seed used when ``order`` is ``RANDOM``.
    """

    order: SteppingOrder = SteppingOrder.ROUND_ROBIN
    allow_simultaneous: bool = False
    max_steps_per_tick: int = 1
    fixed_order: Optional[List[str]] = None
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.order == SteppingOrder.FIXED and self.fixed_order is None:
            raise ValueError("fixed_order must be set when order is FIXED")
        self._rng = np.random.default_rng(self.rng_seed)

    def resolve_order(self, agent_ids: List[str]) -> List[str]:
        """Return the agent ordering for the current tick.

        Args:
            agent_ids: Available agent identifiers.

        Returns:
            Ordered list of agent identifiers.
        """
        if self.order == SteppingOrder.FIXED:
            assert self.fixed_order is not None
            return [a for a in self.fixed_order if a in agent_ids]
        if self.order == SteppingOrder.ROUND_ROBIN:
            return list(agent_ids)
        if self.order == SteppingOrder.RANDOM:
            shuffled = list(agent_ids)
            self._rng.shuffle(shuffled)
            return shuffled
        # TIMING_BASED handled externally by LatencyScheduler
        return list(agent_ids)


# ---------------------------------------------------------------------------
# Agent timing
# ---------------------------------------------------------------------------

@dataclass
class AgentTimingConfig:
    """Per-agent timing parameters (all in seconds).

    Attributes:
        agent_id: Identifier of the agent.
        perception_latency: Time to process a sensor reading.
        compute_latency: Time to compute an action given an observation.
        actuation_latency: Time between issuing an action and it taking
            effect in the environment.
        jitter_std: Standard deviation of Gaussian jitter added to each
            latency component.
    """

    agent_id: str
    perception_latency: float = 0.0
    compute_latency: float = 0.0
    actuation_latency: float = 0.0
    jitter_std: float = 0.0

    @property
    def total_latency(self) -> float:
        """Total end-to-end latency (deterministic component)."""
        return self.perception_latency + self.compute_latency + self.actuation_latency

    def sample_latency(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample total latency with jitter."""
        base = self.total_latency
        if self.jitter_std > 0.0:
            _rng = rng or np.random.default_rng()
            jitter = _rng.normal(0.0, self.jitter_std)
            return max(0.0, base + jitter)
        return base


# ---------------------------------------------------------------------------
# Environment clock
# ---------------------------------------------------------------------------

class EnvironmentClock:
    """Global clock for timestamping environment events.

    The clock can run in *logical* mode (integer ticks) or *wall-clock*
    mode (continuous seconds).  In logical mode every ``advance()`` call
    increments the tick by 1; in wall-clock mode ``advance()`` records the
    real elapsed time.

    Attributes:
        mode: ``"logical"`` or ``"wallclock"``.
    """

    def __init__(self, mode: str = "logical", dt: float = 0.1) -> None:
        if mode not in ("logical", "wallclock"):
            raise ValueError(f"Unknown clock mode: {mode}")
        self.mode = mode
        self.dt = dt
        self._tick: int = 0
        self._time: float = 0.0
        self._wall_start: float = time.monotonic()
        self._event_log: List[Tuple[float, str, Any]] = []

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def current_time(self) -> float:
        if self.mode == "logical":
            return self._tick * self.dt
        return time.monotonic() - self._wall_start

    def advance(self, steps: int = 1) -> float:
        """Advance the clock and return the new time."""
        self._tick += steps
        self._time = self.current_time
        return self._time

    def reset(self) -> None:
        self._tick = 0
        self._time = 0.0
        self._wall_start = time.monotonic()
        self._event_log.clear()

    def record_event(self, label: str, data: Any = None) -> None:
        """Record a timestamped event."""
        self._event_log.append((self.current_time, label, data))

    @property
    def event_log(self) -> List[Tuple[float, str, Any]]:
        return list(self._event_log)

    def __repr__(self) -> str:
        return f"EnvironmentClock(mode={self.mode!r}, tick={self._tick}, time={self.current_time:.4f})"


# ---------------------------------------------------------------------------
# Stale observation model
# ---------------------------------------------------------------------------

class StaleObservationModel:
    """Model observation staleness due to perception latency.

    When an agent has non-zero perception latency, the observation it
    receives may be *stale*: it reflects the environment state from some
    time in the past rather than the current instant.

    This class maintains a buffer of recent states and returns the one
    closest to ``current_time - perception_latency`` for a given agent.
    """

    def __init__(self, buffer_size: int = 256) -> None:
        self._buffer_size = buffer_size
        self._history: List[Tuple[float, EnvironmentState]] = []

    def record(self, timestamp: float, state: EnvironmentState) -> None:
        """Record a state snapshot at the given timestamp."""
        self._history.append((timestamp, state.copy()))
        if len(self._history) > self._buffer_size:
            self._history.pop(0)

    def get_observation_state(
        self,
        current_time: float,
        perception_latency: float,
    ) -> Optional[EnvironmentState]:
        """Return the state observed by an agent with the given latency.

        The returned state corresponds to the most recent snapshot whose
        timestamp is ≤ ``current_time - perception_latency``.
        """
        target = current_time - perception_latency
        best: Optional[EnvironmentState] = None
        best_time = -float("inf")
        for ts, state in self._history:
            if ts <= target and ts > best_time:
                best = state
                best_time = ts
        return best

    def get_staleness(self, current_time: float, perception_latency: float) -> float:
        """Return the observation staleness in seconds."""
        target = current_time - perception_latency
        for ts, _ in reversed(self._history):
            if ts <= target:
                return current_time - ts
        return float("inf")

    def clear(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Abstract multi-agent environment
# ---------------------------------------------------------------------------

class MultiAgentEnv(ABC):
    """Abstract base class for multi-agent environments.

    A multi-agent environment manages a shared world with *N* agents.
    It supports both synchronous stepping (all agents act at once) and
    asynchronous stepping (agents act one at a time, potentially with
    different latencies).

    Subclasses must implement the abstract methods below.  They may also
    override the default observation/action space properties.
    """

    def __init__(
        self,
        agent_ids: Optional[List[str]] = None,
        stepping: Optional[AsyncSteppingSemantics] = None,
        timing_configs: Optional[Dict[str, AgentTimingConfig]] = None,
        clock: Optional[EnvironmentClock] = None,
        stale_obs: Optional[StaleObservationModel] = None,
    ) -> None:
        self._agent_ids: List[str] = agent_ids or []
        self._stepping = stepping or AsyncSteppingSemantics()
        self._timing: Dict[str, AgentTimingConfig] = timing_configs or {}
        self._clock = clock or EnvironmentClock()
        self._stale_obs = stale_obs or StaleObservationModel()
        self._done: bool = False
        self._step_count: int = 0
        self._pending_actions: Dict[str, Any] = {}
        self._last_observations: Dict[str, np.ndarray] = {}

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def _reset_impl(self) -> Dict[str, np.ndarray]:
        """Environment-specific reset logic.

        Returns:
            Mapping from agent id to initial observation.
        """

    @abstractmethod
    def _step_single(
        self, agent_id: str, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one agent's action.

        Args:
            agent_id: The acting agent.
            action: Action array.

        Returns:
            (observation, reward, done, info)
        """

    @abstractmethod
    def _get_state_impl(self) -> EnvironmentState:
        """Return the full environment state."""

    @abstractmethod
    def _set_state_impl(self, state: EnvironmentState) -> None:
        """Restore environment state from *state*."""

    # -- public interface ----------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to an initial state.

        Returns:
            Mapping from agent id to initial observation.
        """
        self._clock.reset()
        self._stale_obs.clear()
        self._done = False
        self._step_count = 0
        self._pending_actions.clear()
        observations = self._reset_impl()
        self._last_observations = dict(observations)
        # Record initial state for staleness model
        self._stale_obs.record(self._clock.current_time, self._get_state_impl())
        return observations

    def step_async(
        self,
        agent_id: str,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute a single agent's action asynchronously.

        The clock advances by one tick and the observation may be stale
        depending on the agent's perception latency.

        Args:
            agent_id: The acting agent.
            action: Action array.

        Returns:
            (observation, reward, done, info)
        """
        if agent_id not in self._agent_ids:
            raise ValueError(f"Unknown agent: {agent_id}")

        self._clock.advance()
        self._clock.record_event("step_async", {"agent": agent_id})

        obs, reward, done, info = self._step_single(agent_id, action)

        # Apply staleness
        timing = self._timing.get(agent_id)
        if timing and timing.perception_latency > 0:
            stale_state = self._stale_obs.get_observation_state(
                self._clock.current_time, timing.perception_latency
            )
            if stale_state is not None:
                info["staleness"] = self._stale_obs.get_staleness(
                    self._clock.current_time, timing.perception_latency
                )

        self._stale_obs.record(self._clock.current_time, self._get_state_impl())
        self._last_observations[agent_id] = obs
        self._step_count += 1
        self._done = done
        return obs, reward, done, info

    def step_sync(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """Execute all agents' actions synchronously.

        All actions are applied to the same state (simultaneous move).
        The clock advances by one tick.

        Args:
            actions: Mapping from agent id to action.

        Returns:
            (observations, rewards, done, info)
        """
        self._clock.advance()
        self._clock.record_event("step_sync", {"agents": list(actions.keys())})

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        done = False
        info: Dict[str, Any] = {}

        order = self._stepping.resolve_order(list(actions.keys()))
        for agent_id in order:
            if agent_id in actions:
                obs, r, d, i = self._step_single(agent_id, actions[agent_id])
                observations[agent_id] = obs
                rewards[agent_id] = r
                done = done or d
                info[agent_id] = i

        self._stale_obs.record(self._clock.current_time, self._get_state_impl())
        self._last_observations.update(observations)
        self._step_count += 1
        self._done = done
        return observations, rewards, done, info

    def get_state(self) -> EnvironmentState:
        """Return a deep copy of the current environment state."""
        return self._get_state_impl().copy()

    def set_state(self, state: EnvironmentState) -> None:
        """Restore environment to *state*."""
        self._set_state_impl(state)

    def get_agent_ids(self) -> List[str]:
        """Return the list of agent identifiers."""
        return list(self._agent_ids)

    @property
    def num_agents(self) -> int:
        return len(self._agent_ids)

    @property
    def clock(self) -> EnvironmentClock:
        return self._clock

    @property
    def done(self) -> bool:
        return self._done

    @property
    def step_count(self) -> int:
        return self._step_count

    # -- spaces (override in subclass) ---------------------------------------

    def observation_space(self, agent_id: str) -> Dict[str, Any]:
        """Return the observation space descriptor for *agent_id*."""
        return {"shape": (), "dtype": "float64", "low": -np.inf, "high": np.inf}

    def action_space(self, agent_id: str) -> Dict[str, Any]:
        """Return the action space descriptor for *agent_id*."""
        return {"shape": (), "dtype": "float64", "low": -np.inf, "high": np.inf}

    # -- utilities -----------------------------------------------------------

    def get_timing_config(self, agent_id: str) -> AgentTimingConfig:
        """Return timing config for *agent_id*, creating a default if needed."""
        if agent_id not in self._timing:
            self._timing[agent_id] = AgentTimingConfig(agent_id=agent_id)
        return self._timing[agent_id]

    def set_timing_config(self, config: AgentTimingConfig) -> None:
        self._timing[config.agent_id] = config

    def last_observation(self, agent_id: str) -> Optional[np.ndarray]:
        return self._last_observations.get(agent_id)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(agents={self._agent_ids}, "
            f"step={self._step_count}, done={self._done})"
        )
